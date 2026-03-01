"""Weekly Strategic Intelligence pipeline: parse, generate, save, trend detection."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any


STRUCTURED_REPORT_PROMPT = """You are generating a Weekly Strategic Intelligence Report.

## Current week sources

### 1. Weekly Reflections
{reflections}

### 2. Energy Log
{energy}

### 3. Projects Log
{projects}

### 4. Opportunity Pipeline
{opportunities}

---

## Previous weeks context (for trend detection)
{previous_weeks}

---

## Instructions

Generate a structured, actionable report in markdown with these sections:

1. **Executive Summary** (2-3 sentences)
   - Key takeaway and recommended focus

2. **Reflections & Insights**
   - Notable reflections, learnings, mindset shifts

3. **Energy Analysis**
   - Patterns, highs, lows, recommendations

4. **Project Status**
   - Progress, blockers, risks, next steps

5. **Opportunity Pipeline**
   - Active opportunities, prioritization, next actions

6. **Trends & Patterns** (compare with previous weeks)
   - Emerging trends, improvements, concerns, momentum shifts

7. **Action Items**
   - Top 3-5 concrete next actions, prioritized

Be concise, specific, and actionable. Use bullet points. Output only the markdown report."""


def load_sources(sync_root: Path) -> dict[str, str]:
    """Load all weekly intelligence sources from sync root."""
    from app.config import get_settings

    settings = get_settings()

    def read_file(p: Path) -> str:
        if not p.exists():
            return ""
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    def read_csv(p: Path) -> str:
        if not p.exists():
            return ""
        try:
            with open(p, encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                rows = list(reader)
            return "\n".join(",".join(row) for row in rows[:300])
        except Exception:
            return ""

    reflections = read_file(sync_root / settings.weekly_reflections_path)
    energy = read_csv(sync_root / settings.energy_log_path)
    projects = read_file(sync_root / settings.projects_log_path)
    opportunities = read_file(sync_root / settings.opportunity_pipeline_path)

    return {
        "reflections": reflections or "(empty)",
        "energy": energy or "(empty)",
        "projects": projects or "(empty)",
        "opportunities": opportunities or "(no opportunity pipeline)",
    }


def load_previous_reports(logs_dir: Path, lookback: int = 4) -> str:
    """Load content of previous weekly reports for trend context."""
    from app.config import get_settings

    reports_dir = logs_dir / get_settings().weekly_reports_dir
    if not reports_dir.exists():
        return "(no previous reports)"

    md_files = sorted(reports_dir.glob("*.md"), reverse=True)
    if not md_files:
        return "(no previous reports)"

    parts = []
    for f in md_files[:lookback]:
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
            parts.append(f"### Week of {f.stem}\n\n{content[:2000]}...")
        except Exception:
            continue

    if not parts:
        return "(no previous reports)"
    return "\n\n---\n\n".join(parts)


def generate_structured_report(
    sources: dict[str, str],
    previous_weeks: str,
) -> str:
    """Generate structured report via LLM."""
    from google import genai

    from app.config import get_settings

    settings = get_settings()
    if not settings.google_api_key:
        return _fallback_report(sources, "GOOGLE_API_KEY not set")

    prompt = STRUCTURED_REPORT_PROMPT.format(
        reflections=sources["reflections"],
        energy=sources["energy"],
        projects=sources["projects"],
        opportunities=sources["opportunities"],
        previous_weeks=previous_weeks,
    )

    try:
        client = genai.Client(api_key=settings.google_api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={"temperature": 0.3},
        )
        text = (response.text or "") if hasattr(response, "text") else ""
        return text.strip() or _fallback_report(sources, "Empty LLM response")
    except Exception as e:
        return _fallback_report(sources, str(e))


def _fallback_report(sources: dict[str, str], reason: str) -> str:
    """Fallback when LLM unavailable."""
    return f"""# Weekly Strategic Intelligence Report

_(Report generation unavailable: {reason})_

## Sources Loaded

### Reflections
{sources['reflections'][:500]}...

### Energy
{sources['energy'][:500]}...

### Projects
{sources['projects'][:500]}...
"""


def save_report(report: str, logs_dir: Path, date_str: str | None = None) -> Path:
    """Save report to logs/weekly_reports/{date}.md."""
    from app.config import get_settings

    reports_dir = logs_dir / get_settings().weekly_reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    date = date_str or datetime.utcnow().strftime("%Y-%m-%d")
    path = reports_dir / f"{date}.md"
    path.write_text(report, encoding="utf-8")
    return path


def detect_trends(report: str, previous_reports: str) -> list[str]:
    """
    Extract trend mentions from the generated report.
    The LLM incorporates trends in section 6; we parse bullet points from that section.
    """
    trends = []
    in_trends = False
    for line in report.split("\n"):
        stripped = line.strip()
        lower = stripped.lower()
        if "trend" in lower and ("##" in stripped or "6." in stripped or "**" in lower):
            in_trends = True
            continue
        if in_trends:
            if "action" in lower or "7." in stripped:
                break
            if stripped.startswith("-") or stripped.startswith("*") or stripped.startswith("•"):
                item = stripped.lstrip("-*• ").strip()
                if item and len(item) > 3:
                    trends.append(item)
    return trends


def run_pipeline(
    sync_root: Path | str | None = None,
    date_str: str | None = None,
) -> dict[str, Any]:
    """
    Run full Weekly Strategic Intelligence pipeline.
    Returns {report, path, trends, sources_loaded}.
    """
    from app.config import get_settings

    settings = get_settings()
    sources_dir = Path(sync_root) if sync_root else _get_sources_dir()
    logs_dir = Path(settings.logs_dir)
    if not logs_dir.is_absolute():
        logs_dir = (Path.cwd() / logs_dir).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    sources = load_sources(sources_dir)
    previous_weeks = load_previous_reports(logs_dir, settings.trend_lookback_weeks)

    report = generate_structured_report(sources, previous_weeks)
    path = save_report(report, logs_dir, date_str)
    trends = detect_trends(report, previous_weeks)

    sources_loaded = {
        k: bool(v and v != "(empty)" and "(no " not in str(v))
        for k, v in sources.items()
    }

    return {
        "report": report,
        "path": str(path),
        "trends": trends,
        "sources_loaded": sources_loaded,
    }


def _get_sources_dir() -> Path:
    """Default directory for weekly sources (local sync of Drive)."""
    from app.config import get_settings

    return Path(get_settings().chroma_persist_path) / "ofir_brain"
