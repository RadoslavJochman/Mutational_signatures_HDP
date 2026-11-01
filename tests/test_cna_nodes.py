"""Tests for realdata/scripts/cna_nodes.py and realdata/scripts/captions.py."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "realdata" / "scripts"))

from captions import upsert_captions  # noqa: E402
from cna_nodes import ancestors, node_roles, parse_scicone_events  # noqa: E402

DIAG = """## SCICoNE node region events
  node 0 (parent NULL): {}
  node 21 (parent 32): {'0': '-1', '10': '1'}
  node 25 (parent 21): {'100': '1'}
  node 30 (parent 32): {'2': '1'}
  node 32 (parent 0): {'222': '1'}
"""


def test_parse_scicone_events(tmp_path):
    p = tmp_path / "diag.txt"
    p.write_text(DIAG)
    nodes = parse_scicone_events(p)
    assert nodes["0"] == {"parent": "NULL", "events": {}}
    assert nodes["21"]["events"] == {"0": "-1", "10": "1"}
    assert nodes["25"]["parent"] == "21"


def test_parse_scicone_events_raises_without_nodes(tmp_path):
    p = tmp_path / "diag.txt"
    p.write_text("nothing here\n")
    with pytest.raises(ValueError):
        parse_scicone_events(p)


def test_ancestors_nearest_first():
    parent = {"0": "NULL", "32": "0", "21": "32", "25": "21"}
    assert ancestors("25", parent) == ["21", "32", "0"]


def test_node_roles_order_and_names():
    parent = {"0": "NULL", "32": "0", "21": "32", "25": "21", "30": "32"}
    n_events = {"21": 56, "25": 7, "30": 1}
    order, names = node_roles(["25", "21", "30"], parent, n_events)
    assert order == ["30", "21", "25"]
    assert names == {"30": "near-diploid", "21": "rearranged clone", "25": "subclone"}


def test_node_roles_rejects_duplicate_names():
    parent = {"0": "NULL", "1": "0", "2": "0", "3": "0"}
    with pytest.raises(ValueError):
        node_roles(["1", "2", "3"], parent, {"1": 0, "2": 5, "3": 6})


def test_upsert_captions_keeps_other_figures(tmp_path):
    p = tmp_path / "captions.md"
    upsert_captions(p, {"Figure 1. A": "one.", "Figure 2. B": "two."})
    upsert_captions(p, {"Figure 6. F": "six."})
    upsert_captions(p, {"Figure 1. A": "uno."})
    text = p.read_text()
    assert text.index("uno.") < text.index("two.") < text.index("six.")
    assert "one." not in text
