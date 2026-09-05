"""Consistency checks for public documentation."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_docs_use_product_facing_language() -> None:
    prohibited = (
        "Reporting" + " Rules",
        "Do not report" + " synthetic",
        "This report is gener" + "ated by",
        "current source" + " tree",
        "live implementation" + " ledger",
        "ROAD" + "MAP.md",
    )
    paths = [ROOT / "README.md"]
    paths.extend(
        path
        for path in (ROOT / "docs").rglob("*.md")
        if "sources" not in path.parts and "_build" not in path.parts
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for phrase in prohibited:
            assert phrase not in text, f"{phrase!r} found in {path.relative_to(ROOT)}"


def test_internal_working_documents_are_absent_from_public_docs() -> None:
    retired = (
        "GPU_VALIDATION",
        "IMPLEMENTATION_STATUS",
        "LITERATURE_LANDSCAPE",
        "REPRODUCIBLE_STUDIES",
        "VALIDATION",
    )
    index = (ROOT / "docs" / "index.rst").read_text(encoding="utf-8")
    for name in retired:
        assert not (ROOT / "docs" / f"{name}.md").exists()
        assert name not in index
    assert not (ROOT / "docs" / "softwarex").exists()


def test_obsolete_planning_page_is_absent() -> None:
    page_name = "ROAD" + "MAP"
    assert not (ROOT / "docs" / f"{page_name}.md").exists()
    assert page_name not in (ROOT / "docs" / "index.rst").read_text(encoding="utf-8")
