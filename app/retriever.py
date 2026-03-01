"""Production hybrid retriever: vector + BM25 + configurable MMR."""

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np

# Supported metadata filter keys for Chroma
METADATA_FILTER_KEYS = {"source_file", "category", "last_updated"}


def _tokenize(text: str) -> list[str]:
    """Tokenize for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


def _rrf_score(ranks: list[int], k: int = 60) -> float:
    """Reciprocal rank fusion score."""
    return sum(1 / (k + r) for r in ranks)


def _normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return []
    arr = np.array(scores, dtype=np.float64)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        return ((arr - lo) / (hi - lo)).tolist()
    return [1.0] * len(scores)


def _mmr_rerank(
    query_embedding: list[float],
    doc_embeddings: list[list[float]],
    doc_ids: list[str],
    doc_contents: list[str],
    doc_metadatas: list[dict[str, Any]],
    hybrid_scores: list[float],
    top_k: int,
    lambda_: float,
) -> list[dict[str, Any]]:
    """
    Apply MMR: MMR = (1-λ)*relevance - λ*max(sim(doc, selected)).
    Returns list of {id, content, metadata, score, ...}.
    """
    if not doc_ids or top_k <= 0:
        return []

    q = np.array(query_embedding, dtype=np.float32)
    embs = np.array(doc_embeddings, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-9)
    embs_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)

    rel_scores = np.dot(embs_norm, q_norm)
    if hybrid_scores:
        rel_norm = np.array(_normalize_scores(hybrid_scores), dtype=np.float64)
        rel_scores = 0.5 * rel_scores + 0.5 * rel_norm

    selected: list[int] = []
    selected_scores: list[float] = []

    for _ in range(min(top_k, len(doc_ids))):
        mmr_scores = np.full(len(doc_ids), -np.inf)
        for i in range(len(doc_ids)):
            if i in selected:
                mmr_scores[i] = -np.inf
                continue
            relevance = float(rel_scores[i])
            max_sim = 0.0
            if selected:
                sims = np.dot(embs_norm[selected], embs_norm[i])
                max_sim = float(np.max(sims))
            mmr = (1 - lambda_) * relevance - lambda_ * max_sim
            mmr_scores[i] = mmr
        best = int(np.argmax(mmr_scores))
        selected.append(best)
        selected_scores.append(float(rel_scores[best]))

    return [
        {
            "id": doc_ids[i],
            "content": doc_contents[i],
            "metadata": doc_metadatas[i] if i < len(doc_metadatas) else {},
            "score": selected_scores[j],
        }
        for j, i in enumerate(selected)
    ]


def _build_chroma_where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build Chroma where clause from API filters. Only allow source_file, category, last_updated."""
    if not filters:
        return None
    where = {k: v for k, v in filters.items() if k in METADATA_FILTER_KEYS}
    return where if where else None


def _metadata_matches(doc_metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Check if document metadata matches all filter conditions."""
    for k, v in filters.items():
        if k not in METADATA_FILTER_KEYS:
            continue
        doc_val = doc_metadata.get(k)
        if doc_val != v:
            return False
    return True


class HybridRetriever:
    """Production hybrid retriever: dense + BM25 + MMR with metadata filtering."""

    def __init__(
        self,
        vector_store: Any,
        embedding_provider: Any,
        bm25_index_path: str,
        mmr_lambda: float = 0.3,
        mmr_fetch_factor: int = 3,
        rrf_k: int = 60,
        vector_weight: float = 0.5,
    ):
        self._vector_store = vector_store
        self._embedding_provider = embedding_provider
        self._bm25_path = Path(bm25_index_path)
        self._mmr_lambda = mmr_lambda
        self._mmr_fetch_factor = mmr_fetch_factor
        self._rrf_k = rrf_k
        self._vector_weight = vector_weight
        self._bm25_weight = 1.0 - vector_weight
        self._bm25_data: dict[str, Any] | None = None

    def _load_bm25(self) -> bool:
        if not self._bm25_path.exists():
            return False
        try:
            with open(self._bm25_path, "rb") as f:
                self._bm25_data = pickle.load(f)
            return True
        except Exception:
            return False

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        debug: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search: vector + BM25, weighted fusion, MMR rerank.
        Supports metadata filtering: source_file, category, last_updated.
        When debug=True, each result includes vector_score, bm25_score, rrf_score.
        """
        chroma_where = _build_chroma_where(filters)
        fetch_n = min(50, max(top_k * self._mmr_fetch_factor, 20))

        # 1. Dense retrieval
        query_emb = self._embedding_provider.embed([query])[0]
        dense_result = self._vector_store.query(
            query_embeddings=[query_emb],
            n_results=fetch_n,
            where=chroma_where,
            include=["documents", "metadatas", "distances"],
        )

        dense_ids = dense_result.get("ids", [[]])[0] or []
        dense_docs = dense_result.get("documents", [[]])[0] or []
        dense_metas = dense_result.get("metadatas", [[]])[0] or []
        dense_distances = dense_result.get("distances", [[]])[0] or []
        dense_similarities = [1 / (1 + (d or 0)) for d in dense_distances]

        # 2. BM25 retrieval (with metadata filter applied post-hoc)
        bm25_ids: list[str] = []
        bm25_docs: list[str] = []
        bm25_metas: list[dict] = []
        bm25_scores: list[float] = []

        if self._load_bm25() and self._bm25_data:
            bm25 = self._bm25_data["bm25"]
            ids = self._bm25_data["ids"]
            docs = self._bm25_data["docs"]
            metas = self._bm25_data.get("metadatas") or []
            tokenized_query = _tokenize(query)
            if tokenized_query and docs:
                raw_scores = bm25.get_scores(tokenized_query)
                top_indices = np.argsort(raw_scores)[::-1][: fetch_n]
                for idx in top_indices:
                    if raw_scores[idx] <= 0:
                        continue
                    meta = metas[idx] if idx < len(metas) else {}
                    if filters and not _metadata_matches(meta, filters):
                        continue
                    bm25_ids.append(ids[idx])
                    bm25_docs.append(docs[idx] if docs[idx] else "")
                    bm25_metas.append(meta)
                    bm25_scores.append(float(raw_scores[idx]))

        # 3. Build fused pool with hybrid scores
        id_to_info: dict[str, dict[str, Any]] = {}
        for i, cid in enumerate(dense_ids):
            dense_rank = i + 1
            vs = dense_similarities[i] if i < len(dense_similarities) else 0.0
            if cid not in id_to_info:
                id_to_info[cid] = {
                    "id": cid,
                    "content": dense_docs[i] if i < len(dense_docs) else "",
                    "metadata": dense_metas[i] if i < len(dense_metas) else {},
                    "dense_rank": dense_rank,
                    "bm25_rank": 999,
                    "vector_score": vs,
                    "bm25_score": 0.0,
                }
            else:
                id_to_info[cid]["dense_rank"] = min(id_to_info[cid]["dense_rank"], dense_rank)
                id_to_info[cid]["vector_score"] = max(id_to_info[cid].get("vector_score", 0), vs)

        for i, cid in enumerate(bm25_ids):
            bm25_rank = i + 1
            bs = bm25_scores[i] if i < len(bm25_scores) else 0.0
            if cid in id_to_info:
                id_to_info[cid]["bm25_rank"] = min(id_to_info[cid].get("bm25_rank", 999), bm25_rank)
                id_to_info[cid]["bm25_score"] = max(id_to_info[cid].get("bm25_score", 0), bs)
                if i < len(bm25_docs):
                    id_to_info[cid]["content"] = id_to_info[cid].get("content") or bm25_docs[i]
                if i < len(bm25_metas) and bm25_metas[i]:
                    id_to_info[cid]["metadata"].update(bm25_metas[i])
            else:
                id_to_info[cid] = {
                    "id": cid,
                    "content": bm25_docs[i] if i < len(bm25_docs) else "",
                    "metadata": bm25_metas[i] if i < len(bm25_metas) else {},
                    "dense_rank": 999,
                    "bm25_rank": bm25_rank,
                    "vector_score": 0.0,
                    "bm25_score": bs,
                }

        # 4. Hybrid score: weighted linear combination of normalized vector + BM25
        all_vector = [info["vector_score"] for info in id_to_info.values()]
        all_bm25 = [info["bm25_score"] for info in id_to_info.values()]
        norm_vector = _normalize_scores(all_vector)
        norm_bm25 = _normalize_scores(all_bm25)
        for info, nv, nb in zip(id_to_info.values(), norm_vector, norm_bm25):
            info["hybrid_score"] = self._vector_weight * nv + self._bm25_weight * nb
            info["rrf_score"] = _rrf_score([info["dense_rank"], info["bm25_rank"]], k=self._rrf_k)

        pool = sorted(id_to_info.values(), key=lambda x: -x["hybrid_score"])[: fetch_n]

        if not pool:
            return []

        # 5. MMR reranking
        pool_ids = [p["id"] for p in pool]
        pool_contents = [p["content"] for p in pool]
        pool_metas = [p["metadata"] for p in pool]
        pool_hybrid = [p["hybrid_score"] for p in pool]
        pool_embeddings = self._embedding_provider.embed(pool_contents)

        mmr_results = _mmr_rerank(
            query_embedding=query_emb,
            doc_embeddings=pool_embeddings,
            doc_ids=pool_ids,
            doc_contents=pool_contents,
            doc_metadatas=pool_metas,
            hybrid_scores=pool_hybrid,
            top_k=top_k,
            lambda_=self._mmr_lambda,
        )

        # Build id -> debug info for results
        pool_by_id = {p["id"]: p for p in pool}

        out: list[dict[str, Any]] = []
        for r in mmr_results:
            item: dict[str, Any] = {
                "content": r["content"],
                "metadata": r["metadata"],
                "score": r["score"],
                "chunk_id": r["id"],
            }
            if debug:
                info = pool_by_id.get(r["id"], {})
                item["scores"] = {
                    "vector_score": round(info.get("vector_score", 0), 4),
                    "bm25_score": round(info.get("bm25_score", 0), 4),
                    "rrf_score": round(info.get("rrf_score", 0), 4),
                    "hybrid_score": round(info.get("hybrid_score", 0), 4),
                    "mmr_score": round(r["score"], 4),
                }
            out.append(item)

        return out


def get_retriever() -> HybridRetriever:
    """Factory for hybrid retriever."""
    from app.config import get_settings
    from app.embeddings import get_embedding_provider
    from app.vector_store import get_vector_store

    settings = get_settings()
    vs = get_vector_store()
    emb = get_embedding_provider()
    bm25_path = str(settings.resolve_chroma_path() / "bm25_index.pkl")
    return HybridRetriever(
        vector_store=vs,
        embedding_provider=emb,
        bm25_index_path=bm25_path,
        mmr_lambda=settings.mmr_lambda,
        mmr_fetch_factor=settings.mmr_fetch_factor,
        rrf_k=settings.rrf_k,
        vector_weight=settings.vector_weight,
    )
