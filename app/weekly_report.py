"""Weekly report generator from reflections, energy log, and projects log."""

import csv
from pathlib import Path
from typing import Any


WEEKLY_REPORT_PROMPT = """You are summarizing a weekly personal review for an executive summary.

Below are three sources:
1. Weekly reflections
2. Energy log (CSV)
3. Projects log

Create a concise executive summary (2-4 paragraphs) that covers:
- Key reflections and insights from the week
- Energy patterns and notable highs/lows
- Project progress and blockers

Write in clear, actionable prose. Output markdown."""


def load_weekly_sources(sync_root: Path) -> dict[str, str]:
    """
    Load weekly_reflections.md, energy_log.csv, projects_log.md from sync_root.
    Returns dict with keys: reflections, energy, projects.
    """
    from app.config import get_settings

    settings = get_settings()
    reflections_path = sync_root / settings.weekly_reflections_path
    energy_path = sync_root / settings.energy_log_path
    projects_path = sync_root / settings.projects_log_path

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
            return "\n".join(",".join(row) for row in rows[:200])
        except Exception:
            return ""

    return {
        "reflections": read_file(reflections_path),
        "energy": read_csv(energy_path),
        "projects": read_file(projects_path),
    }


def generate_weekly_report(
    sync_root: Path | str | None = None,
) -> str:
    """
    Load weekly sources, concatenate, call LLM, return markdown executive summary.
    """
    from google import genai

    from app.config import get_settings

    settings = get_settings()
    sources_dir = Path(sync_root) if sync_root else _get_weekly_sources_dir()
    sources = load_weekly_sources(sources_dir)

    combined = f"""
## Weekly Reflections
{sources['reflections'] or '(empty)'}

## Energy Log
{sources['energy'] or '(empty)'}

## Projects Log
{sources['projects'] or '(empty)'}
""".strip()

    if not settings.google_api_key:
        return f"# Weekly Report\n\n(Sources loaded; GOOGLE_API_KEY not set for LLM summary)\n\n{combined}"

    client = genai.Client(api_key=settings.google_api_key)
    full_prompt = f"{WEEKLY_REPORT_PROMPT}\n\n---\n\n{combined}"
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=full_prompt,
        config={"temperature": 0.3},
    )
    return (response.text or "") if hasattr(response, "text") else "# Weekly Report\n\n(No summary generated)"


def _get_weekly_sources_dir() -> Path:
    """Default directory for weekly report source files (local sync of Drive)."""
    from app.config import get_settings

    settings = get_settings()
    # Default: ./db/ofir_brain - Drive sync could write files here
    return Path(settings.chroma_persist_path) / "ofir_brain"
