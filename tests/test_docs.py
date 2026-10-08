"""Every relative link in the documentation and the website must point at a real file."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    ROOT / "README.md",
    ROOT / "model_card.md",
    *sorted((ROOT / "docs").glob("*.md")),
    *sorted((ROOT / "results").rglob("*.md")),
    ROOT / "site" / "index.html",
]
LINK = re.compile(r"""(?:\]\(|src=["']|href=["'])([^)"'#\s]+)""")


def _relative_links(path: Path) -> list[str]:
    links = LINK.findall(path.read_text(encoding="utf-8"))
    return [x for x in links if not re.match(r"^(https?:|mailto:|data:|#)", x)]


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(doc):
    missing = [x for x in _relative_links(doc) if not (doc.parent / x).exists()]
    assert not missing, f"{doc.name} links to missing paths: {missing}"


def test_site_data_is_generated_from_results():
    """site/data.js must be exactly what site/build_data.py produces from results/."""
    import runpy

    data = ROOT / "site" / "data.js"
    before = data.read_bytes()
    runpy.run_path(str(ROOT / "site" / "build_data.py"), run_name="__main__")
    assert data.read_bytes() == before, "site/data.js is stale: run python site/build_data.py"
