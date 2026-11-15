"""Assemblage de pages HTML autonomes (Plotly embarqué, ouvrables hors-ligne)."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

from transilien.utils import get_logger

log = get_logger("reporting")

CSS = """
:root { --fg:#1d2433; --muted:#5b6475; --bg:#fafbfc; --card:#fff; --line:#e3e6ec; --accent:#0b5cad; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6e9ef; --muted:#9aa3b2; --bg:#12151b; --card:#1b1f27; --line:#2c323d; --accent:#6aa9ff; }
}
* { box-sizing: border-box; }
body { margin:0; font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; color:var(--fg); background:var(--bg); }
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; } h2 { font-size: 19px; margin: 36px 0 12px; border-bottom:1px solid var(--line); padding-bottom:6px; }
.sub { color: var(--muted); margin: 0 0 20px; }
.kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:12px; }
.kpi { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }
.kpi .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
.kpi .l { color: var(--muted); font-size: 13px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; font-size: 14px; }
th, td { text-align: left; padding: 5px 10px; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px; margin-bottom:16px; overflow-x:auto; }
.grid2 { display:grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap:16px; }
.note { color: var(--muted); font-size: 13px; }
.good { color: #1a7f37; } .bad { color: #cf222e; }
"""


@dataclass
class Page:
    title: str
    subtitle: str = ""
    parts: list[str] = field(default_factory=list)
    figures: dict[str, go.Figure] = field(default_factory=dict)
    _plotly_embedded: bool = False

    def h2(self, text: str) -> None:
        self.parts.append(f"<h2>{html.escape(text)}</h2>")

    def raw(self, fragment: str) -> None:
        self.parts.append(fragment)

    def note(self, text: str) -> None:
        self.parts.append(f'<p class="note">{html.escape(text)}</p>')

    def kpis(self, items: list[tuple[str, str]]) -> None:
        cells = "".join(
            f'<div class="kpi"><div class="v">{html.escape(v)}</div><div class="l">{html.escape(k)}</div></div>'
            for k, v in items
        )
        self.parts.append(f'<div class="kpis">{cells}</div>')

    def table(self, df: pd.DataFrame, float_format: str = "{:.4f}", index: bool = True) -> None:
        formatters = {c: float_format.format for c in df.select_dtypes("float").columns}
        table_html = df.to_html(
            index=index, border=0, formatters=formatters, na_rep="—", escape=True
        )
        self.parts.append(f'<div class="card">{table_html}</div>')

    def figure(self, key: str, fig: go.Figure) -> None:
        include = "inline" if not self._plotly_embedded else False
        self._plotly_embedded = True
        self.figures[key] = fig
        fig.update_layout(template="plotly_white", margin={"l": 50, "r": 20, "t": 50, "b": 45})
        self.parts.append(
            f'<div class="card">{fig.to_html(full_html=False, include_plotlyjs=include)}</div>'
        )

    def render(self) -> str:
        body = "\n".join(self.parts)
        return (
            f'<!doctype html><html lang="fr"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{html.escape(self.title)}</title><style>{CSS}</style></head>"
            f'<body><main><h1>{html.escape(self.title)}</h1><p class="sub">{html.escape(self.subtitle)}</p>'
            f"{body}</main></body></html>"
        )

    def save(self, path: Path, export_png_dir: Path | None = None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(), encoding="utf-8")
        if export_png_dir is not None:
            export_png(self.figures, export_png_dir)
        return path


def export_png(figures: dict[str, go.Figure], out_dir: Path) -> None:
    """Export PNG au mieux (nécessite l'extra ``export`` / kaleido) ; n'échoue jamais."""
    try:
        import kaleido  # noqa: F401
    except ImportError:
        log.info("kaleido absent : export PNG ignoré (uv sync --extra export).")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, fig in figures.items():
        try:
            fig.write_image(out_dir / f"{key}.png", width=1100, height=550, scale=1)
        except Exception as exc:
            first_line = (
                str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
            )
            log.warning(
                "Export PNG ignoré (%s). Kaleido ≥ 1 requiert Chrome : `uv run plotly_get_chrome` "
                "ou reporting.export_png=false.",
                first_line,
            )
            return
