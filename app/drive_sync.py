"""Google Drive sync with hash-based change detection."""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tiktoken

# Google Drive MIME types for text export
MIME_EXPORT = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}

TEXT_MIMES = {
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.google-apps.presentation",
}


@dataclass
class FileInfo:
    """Metadata for a Drive file."""

    id: str
    name: str
    mime_type: str
    modified_time: str
    content_hash: str | None
    content: str | None = None
    parent_path: str = ""


@dataclass
class SyncResult:
    """Result of a sync operation."""

    changed_files: list[FileInfo] = field(default_factory=list)
    all_files: list[FileInfo] = field(default_factory=list)
    indexed: int = 0
    skipped: int = 0


def _get_drive_service():
    """Build Google Drive API service. Supports Service Account and OAuth installed app."""
    from google.oauth2 import service_account
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build

    from app.config import get_settings

    settings = get_settings()
    creds = None
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

    def _get_creds_from_file(path: str):
        p = Path(path)
        if not p.is_absolute():
            p = Path.cwd() / path
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    if settings.google_drive_credentials_json:
        creds_dict = json.loads(settings.google_drive_credentials_json)
    elif settings.google_application_credentials:
        creds_dict = _get_creds_from_file(settings.google_application_credentials)
    else:
        creds_dict = None

    if creds_dict:
        if creds_dict.get("type") == "service_account":
            creds = service_account.Credentials.from_service_account_info(
                creds_dict, scopes=SCOPES
            )
        elif "installed" in creds_dict or "web" in creds_dict:
            client_config = creds_dict.get("installed") or creds_dict.get("web", {})
            flow = InstalledAppFlow.from_client_config(creds_dict, SCOPES)
            token_path = Path(settings.chroma_persist_path) / "drive_token.json"
            if token_path.exists():
                from google.oauth2.credentials import Credentials
                token_data = json.loads(token_path.read_text())
                creds = Credentials(
                    token=token_data.get("token"),
                    refresh_token=token_data.get("refresh_token"),
                    token_uri="https://oauth2.googleapis.com/token",
                    client_id=client_config.get("client_id"),
                    client_secret=client_config.get("client_secret"),
                    scopes=SCOPES,
                )
                if creds.expired and creds.refresh_token:
                    creds.refresh(Request())
            else:
                creds = flow.run_local_server(port=0)
                token_path.parent.mkdir(parents=True, exist_ok=True)
                token_path.write_text(json.dumps({
                    "token": creds.token,
                    "refresh_token": creds.refresh_token,
                }, indent=2))
        else:
            raise ValueError("credentials.json: Expected 'type: service_account' or 'installed' key")
    else:
        import google.auth
        creds, _ = google.auth.default(scopes=SCOPES)

    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _list_files_recursive(service: Any, folder_id: str, parent_path: str = "") -> list[FileInfo]:
    """List all files under a folder recursively."""
    files: list[FileInfo] = []
    page_token = None

    while True:
        query = f"'{folder_id}' in parents and trashed = false"
        results = (
            service.files()
            .list(
                q=query,
                pageSize=100,
                fields="nextPageToken, files(id, name, mimeType, modifiedTime, md5Checksum)",
                pageToken=page_token,
            )
            .execute()
        )
        items = results.get("files", [])
        page_token = results.get("nextPageToken")

        for item in items:
            mime = item.get("mimeType", "")
            fid = item["id"]
            name = item.get("name", "")
            modified = item.get("modifiedTime", "")
            md5 = item.get("md5Checksum")

            # If it's a folder, recurse
            if mime == "application/vnd.google-apps.folder":
                subpath = f"{parent_path}/{name}".lstrip("/")
                sub_files = _list_files_recursive(service, fid, subpath)
                files.extend(sub_files)
                continue

            # For binary files, use md5 if available
            content_hash = md5 if md5 else None
            fi = FileInfo(
                id=fid,
                name=name,
                mime_type=mime,
                modified_time=modified,
                content_hash=content_hash,
                parent_path=parent_path or ".",
            )
            files.append(fi)

        if not page_token:
            break

    return files


def _download_and_hash(service: Any, file_info: FileInfo) -> tuple[str, str]:
    """
    Download file content and compute hash. Returns (content, hash).
    For native Google files, exports to text and hashes that.
    """
    fid = file_info.id
    mime = file_info.mime_type

    if mime in MIME_EXPORT:
        # Export native Google docs to text
        export_mime = MIME_EXPORT[mime]
        try:
            content = (
                service.files()
                .export_media(fileId=fid, mimeType=export_mime)
                .execute()
            )
            if isinstance(content, bytes):
                text = content.decode("utf-8", errors="replace")
            else:
                text = str(content)
        except Exception:
            text = ""
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return text, content_hash

    if mime in TEXT_MIMES or mime.startswith("text/"):
        try:
            content = service.files().get_media(fileId=fid).execute()
            if isinstance(content, bytes):
                text = content.decode("utf-8", errors="replace")
            else:
                text = str(content)
        except Exception:
            text = ""
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return text, content_hash

    # Binary: use md5 from API if we have it, else download and hash
    try:
        meta = service.files().get(fileId=fid, fields="md5Checksum").execute()
        md5 = meta.get("md5Checksum")
        if md5:
            content = service.files().get_media(fileId=fid).execute()
            return "", md5  # Don't index binary content
    except Exception:
        pass
    return "", ""


def _get_sync_state_path(persist_path: str) -> Path:
    return Path(persist_path) / "sync_state.json"


def _load_sync_state(persist_path: str) -> dict[str, dict[str, str]]:
    path = _get_sync_state_path(persist_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_sync_state(persist_path: str, state: dict[str, dict[str, str]]) -> None:
    path = _get_sync_state_path(persist_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into ~chunk_size token chunks with overlap."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        if end >= len(tokens):
            break
        start = end - overlap
    return chunks


def _tokenize_for_bm25(text: str) -> list[str]:
    """Simple tokenization for BM25 (lowercase, split on non-alnum)."""
    return re.findall(r"\b\w+\b", text.lower())


def get_changed_files(
    service: Any, folder_id: str, persist_path: str
) -> tuple[list[FileInfo], list[FileInfo]]:
    """
    List all files under folder_id and return (changed_files, all_files).
    Changed = new or hash different from sync_state.
    """
    state = _load_sync_state(persist_path)
    all_files = _list_files_recursive(service, folder_id)
    changed: list[FileInfo] = []

    for fi in all_files:
        if fi.mime_type == "application/vnd.google-apps.folder":
            continue
        if fi.mime_type not in TEXT_MIMES and fi.mime_type not in MIME_EXPORT:
            if not fi.content_hash:
                continue
        stored = state.get(fi.id, {})
        stored_hash = stored.get("hash")
        if stored_hash is None:
            changed.append(fi)
            continue
        if fi.content_hash is None:
            _, content_hash = _download_and_hash(service, fi)
            fi.content_hash = content_hash
            if content_hash != stored_hash:
                changed.append(fi)
        else:
            if fi.content_hash != stored_hash:
                changed.append(fi)

    return changed, all_files


def sync_folder(
    folder_id: str | None = None,
    persist_path: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> SyncResult:
    """
    Sync /ofir_brain from Drive, detect changes, and return files to index.
    Does NOT write to vector store; caller (reindex) does that.
    """
    from app.config import get_settings
    from app.embeddings import get_embedding_provider
    from app.vector_store import get_vector_store

    settings = get_settings()
    fid = folder_id or settings.ofir_brain_folder_id
    path = persist_path or str(settings.resolve_chroma_path())
    cs = chunk_size or settings.chunk_size
    co = chunk_overlap or settings.chunk_overlap

    if not fid:
        return SyncResult(indexed=0, skipped=0)

    service = _get_drive_service()
    state = _load_sync_state(path)
    changed, all_files = get_changed_files(service, fid, path)

    # Now download content for changed files and index
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store(path)

    # Build list of chunk IDs to delete for changed files
    existing_ids = vector_store.get_all_ids()
    to_delete = [cid for cid in existing_ids if any(cid.startswith(f"{fi.id}_") for fi in changed)]
    if to_delete:
        vector_store.delete(to_delete)

    ids_to_add: list[str] = []
    docs_to_add: list[str] = []
    metas_to_add: list[dict[str, Any]] = []

    indexed = 0
    for fi in changed:
        content, content_hash = _download_and_hash(service, fi)
        if not content.strip():
            state[fi.id] = {"hash": content_hash, "modified": fi.modified_time}
            continue
        chunks = _chunk_text(content, cs, co)
        rel_path = f"{fi.parent_path}/{fi.name}".strip("/") if fi.parent_path else fi.name
        for i, chunk in enumerate(chunks):
            cid = f"{fi.id}_{i}"
            ids_to_add.append(cid)
            docs_to_add.append(chunk)
        metas_to_add.append({
            "source_file": rel_path,
            "category": _infer_category(rel_path),
            "last_updated": fi.modified_time[:10] if fi.modified_time else "",
            "file_id": fi.id,
            "content_hash": content_hash,
        })
        state[fi.id] = {"hash": content_hash, "modified": fi.modified_time}
        # Persist to local dir for weekly report (weekly_reflections, energy_log, projects_log)
        _write_local_copy(path, fi, content)
        indexed += 1

    if ids_to_add:
        embeddings = embedding_provider.embed(docs_to_add)
        vector_store.add(ids_to_add, docs_to_add, metas_to_add, embeddings)

    _save_sync_state(path, state)

    # Rebuild BM25 index (caller or we do it here)
    _rebuild_bm25_index(vector_store, path)

    skipped = len(all_files) - len(changed)
    return SyncResult(
        changed_files=changed,
        all_files=all_files,
        indexed=indexed,
        skipped=skipped,
    )


def _write_local_copy(persist_path: str, fi: FileInfo, content: str) -> None:
    """Write file content to local mirror for weekly report sources."""
    if not content:
        return
    root = Path(persist_path) / "ofir_brain"
    rel = f"{fi.parent_path}/{fi.name}".strip("/") if fi.parent_path else fi.name
    local_path = root / rel
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        local_path.write_text(content, encoding="utf-8")
    except Exception:
        pass


def _infer_category(rel_path: str) -> str:
    """Infer category from file path."""
    path_lower = rel_path.lower()
    if "reflection" in path_lower or "weekly" in path_lower:
        return "reflections"
    if "energy" in path_lower:
        return "energy"
    if "project" in path_lower:
        return "projects"
    return "general"


def _rebuild_bm25_index(vector_store: Any, persist_path: str) -> None:
    """Rebuild BM25 index from vector store and persist to db/bm25_index.pkl."""
    import pickle

    from rank_bm25 import BM25Okapi

    ids, docs, metas = vector_store.get_all_documents()
    if not docs:
        return
    tokenized = [_tokenize_for_bm25(str(d) if d else "") for d in docs]
    bm25 = BM25Okapi(tokenized)
    index_path = Path(persist_path) / "bm25_index.pkl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids, "docs": docs, "metadatas": metas or []}, f)


def full_reindex(
    folder_id: str | None = None,
    persist_path: str | None = None,
) -> SyncResult:
    """
    Full reindex: clear existing state, re-sync everything.
    """
    from app.config import get_settings
    from app.vector_store import VectorStore, get_vector_store

    settings = get_settings()
    fid = folder_id or settings.ofir_brain_folder_id
    path = persist_path or str(settings.resolve_chroma_path())

    # Clear sync state to force re-index of all files
    state_path = _get_sync_state_path(path)
    if state_path.exists():
        state_path.unlink()

    # Clear vector store collection and rebuild
    vs = get_vector_store(path)
    all_ids = vs.get_all_ids()
    if all_ids:
        vs.delete(all_ids)

    return sync_folder(folder_id=fid, persist_path=path)
