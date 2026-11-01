"""SCICoNE node helpers shared by the stage 09 consumers: parsing the node
events out of ``cna_tree_diagnostics.txt`` and naming the occupied nodes by
their place in the tree.

Standard library only, so the Euler-side scripts can import it without the
plotting stack.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_NODE_LINE = re.compile(r"^\s*node (\S+) \(parent (\S+)\): (\{.*\})\s*$")


def parse_scicone_events(path: Path) -> dict[str, dict]:
    """SCICoNE node -> ``{'parent': str, 'events': {region: delta}}`` from
    the stage 09 diagnostics' 'SCICoNE node region events' section. The root's
    parent is the string ``'NULL'``. Region keys and deltas are kept as SCICoNE
    wrote them (strings); callers convert. Raises ValueError if no node line
    is found."""
    nodes = {}
    for line in Path(path).read_text().splitlines():
        m = _NODE_LINE.match(line)
        if m:
            nodes[m.group(1)] = {
                "parent": m.group(2),
                "events": ast.literal_eval(m.group(3)),
            }
    if not nodes:
        raise ValueError(f"no SCICoNE node events found in {path}")
    return nodes


def ancestors(node: str, parent: dict[str, str]) -> list[str]:
    """Ancestors of ``node``, nearest first, following ``parent`` until a
    node whose parent is ``'NULL'`` or absent."""
    out = []
    while parent.get(node, "NULL") != "NULL":
        node = parent[node]
        out.append(node)
    return out


def node_roles(
    occupied: list[str], parent: dict[str, str], n_events: dict[str, int]
) -> tuple[list[str], dict[str, str]]:
    """Tree order and descriptive names for the occupied SCICoNE nodes.

    The near-diploid node is the occupied node with the fewest region events.
    Every other occupied node is a "subclone" if one of its SCICoNE ancestors
    is another occupied non-near-diploid node, else the "rearranged clone".
    The order is near-diploid first, then the rest by SCICoNE depth (ties by
    numeric id). Returns ``(order, {node: name})``; raises ValueError if two
    nodes would share a name, since the figures then cannot tell them apart.
    """
    near_diploid = min(occupied, key=lambda n: (n_events[n], int(n)))
    tumour = [n for n in occupied if n != near_diploid]
    tumour.sort(key=lambda n: (len(ancestors(n, parent)), int(n)))
    names = {near_diploid: "near-diploid"}
    for node in tumour:
        nested = any(a in tumour for a in ancestors(node, parent))
        names[node] = "subclone" if nested else "rearranged clone"
    if len(set(names.values())) != len(names):
        raise ValueError(f"occupied SCICoNE nodes do not get unique names: {names}")
    return [near_diploid] + tumour, names
