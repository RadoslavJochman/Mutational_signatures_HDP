"""Shared captions.md writer for the slice D figure scripts.

Each figure script owns some of the captions. ``upsert_captions`` replaces or
adds only those, keeping every other figure's caption, so running one script
never drops another's paragraph. Paragraphs are ``**Figure N. Title.** body``
and are kept in figure-number order under one heading.
"""

from __future__ import annotations

import re
from pathlib import Path

HEADING = "# Slice D meeting figures"
_PARAGRAPH = re.compile(r"^\*\*(Figure (\d+)\.[^*]*?)\.\*\* ", re.S)


def upsert_captions(path: Path, captions: dict[str, str]) -> None:
    """Write ``captions`` (title -> body, title like ``'Figure 6. CNV
    heatmap'``) into ``path``, replacing paragraphs with the same figure
    number and keeping the rest. Raises ValueError for a title with no figure
    number."""
    path = Path(path)
    paragraphs: dict[int, str] = {}
    if path.exists():
        for block in path.read_text().split("\n\n"):
            m = _PARAGRAPH.match(block.strip())
            if m:
                paragraphs[int(m.group(2))] = block.strip()
    for title, body in captions.items():
        m = re.match(r"Figure (\d+)\.", title)
        if not m:
            raise ValueError(f"caption title has no figure number: {title!r}")
        paragraphs[int(m.group(1))] = f"**{title}.** {body}"
    text = HEADING + "\n\n" + "\n\n".join(paragraphs[k] for k in sorted(paragraphs))
    path.write_text(text + "\n")
