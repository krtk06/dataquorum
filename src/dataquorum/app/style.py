"""Shared design tokens and layout rules for the Studio workbench."""

from pathlib import Path

BACKGROUND = "#171715"
SURFACE = "#20201d"
BORDER = "#3c3a34"
INK = "#edeae3"
MUTED = "#b4afa4"
ACCENT = "#c6af86"
CHART = "#c3bfb4"

THEME = {
    "base": "dark",
    "primaryColor": ACCENT,
    "backgroundColor": BACKGROUND,
    "secondaryBackgroundColor": SURFACE,
    "textColor": INK,
    "borderColor": BORDER,
    "baseRadius": "0.5rem",
    "buttonRadius": "0.375rem",
    "showWidgetBorder": True,
}

STYLE = f"""<style>
:root {{
  --dq-ground: {BACKGROUND};
  --dq-surface: {SURFACE};
  --dq-rule: {BORDER};
  --dq-ink: {INK};
  --dq-muted: {MUTED};
  --dq-action: {ACCENT};
}}
</style>
<style>{Path(__file__).with_name("studio.css").read_text(encoding="utf-8")}</style>"""

BRAND = """
<div class="dq-brand">
  <span class="dq-mark" aria-hidden="true"><i></i><i></i><i></i></span>
  <strong>DataQuorum</strong><span class="dq-brand-studio">Studio</span>
</div>
"""
