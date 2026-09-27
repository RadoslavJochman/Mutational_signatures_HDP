"""Check that stage 8/9's real-data tree inputs actually load the way
``TreeHDP`` loads them, before trusting any inference run on them.

This is a one-off verification tool, not part of the generate-infer-score
pipeline (see the "Diagnostic scripts" section of CLAUDE.md for the
precedent: interpretation goes to stdout here, not a CSV).

Reuses the model's own loading code rather than reimplementing it: every
tree is loaded by actually constructing a ``TreeHDP`` (its ``__init__``
parses the Newick string with phylox, relabels nodes to their Newick
labels, and composes multi-tree forests into one graph -- exactly the
code path a real inference run takes), then this script only INSPECTS
the resulting ``model.graph`` and re-runs the same "is this node
observed" test ``_build_pymc_model`` uses internally.

DATA-LOADING CONTRACT this script exists to check, and enforces on its own
input (crossing the realdata/src import boundary CLAUDE.md otherwise keeps
strict -- see the module's own note below): a data-matrix CSV's index must
be read as ``str``. Newick node labels are always Python ``str`` (phylox
parses every leaf/internal label as text), but SECEDO cluster IDs are
small integers, so a plain ``pd.read_csv(path, index_col=0)`` infers an
``int64`` index from a file like ``spectra.csv`` -- pandas has no way to
know the column is meant to line up with string tree labels. The mismatch
raises nothing: ``label in data_matrix.index`` is simply ``False`` for
every node, ``_build_pymc_model`` finds no observed nodes at all, skips
building the ``Multinomial`` likelihood entirely, and the model happily
samples from the prior alone. ``read_data_matrix`` below is the fix
(``index.astype(str)`` after loading); ``run_inference.py``'s own
``pd.read_csv(data_cfg["count_matrix"], index_col=0)`` does not have this
problem today only because the simulator's own node names
(``T1_1``, ...) are never all-digit strings, so pandas never infers a
numeric index for them. A real-data run over SECEDO's numeric cluster IDs
needs this fix wherever a count matrix is loaded, not just here.

Usage
-----
    python realdata/scripts/check_tree_inputs.py \\
        --tree-dir realdata/local_tree_input \\
        --cosmic-signatures COSMIC_sig/cosmic_signatures.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.analysis import DEFAULT_PRIORS  # noqa: E402
from src.models.hdp_inference import TreeHDP  # noqa: E402

GERMLINE_ROOT_ID = "germline"
HIDDEN_PATTERN = re.compile(r"^g\d+$")

# A conservative consensus tree (no attempt to resolve which of 9/10 joins
# {7,8} first -- see build_snv_tree.py's Dollo runner-up): built here, not
# read from a file, since it is a hypothetical to check the loader against,
# not a pipeline output.
CONSENSUS_NEWICK = "(3,(7,8,9,10)g0)germline;"


def read_data_matrix(path: Path) -> pd.DataFrame:
    """Read a count/spectra CSV with its index forced to ``str``.

    See the module docstring: a plain ``pd.read_csv(index_col=0)`` infers
    ``int64`` for an all-numeric index like SECEDO's cluster IDs, which
    then matches no Newick label (always ``str``) and silently drops every
    node from the likelihood. This is the fix, applied once here rather
    than left to each caller to remember.
    """
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df


def describe_tree(model: TreeHDP) -> pd.DataFrame:
    """One row per node in ``model.graph``: its parent (``None`` for the
    root), whether ``_build_pymc_model`` would treat it as observed (in
    ``data_matrix.index`` AND its row sums to more than zero -- the exact
    test the model itself runs), and whether it is the root.
    """
    data_index = set(model.data_matrix.index.astype(str))
    rows = []
    for node in model.graph.nodes():
        parents = list(model.graph.predecessors(node))
        parent = parents[0] if parents else None
        observed = (
            node in data_index
            and float(model.data_matrix.loc[node].to_numpy().sum()) > 0
        )
        rows.append(
            {
                "node": node,
                "parent": parent,
                "observed": observed,
                "is_root": parent is None,
            }
        )
    return pd.DataFrame(rows).set_index("node")


def assert_tree_contract(
    desc: pd.DataFrame,
    expected_observed: Iterable[str],
    germline_id: str = GERMLINE_ROOT_ID,
    hidden_pattern: re.Pattern = HIDDEN_PATTERN,
    excluded_ids: Iterable[str] = (),
) -> None:
    """Raise ``AssertionError`` with a specific message if the tree does
    not satisfy the contract ``_BaseTreeHDP``/``TreeHDP`` needs:

    - exactly one root, and it is ``germline_id``;
    - the root is latent (no spectrum);
    - the observed node set is exactly ``expected_observed``;
    - every other node matches ``hidden_pattern`` (a hidden ``g<k>`` group
      node);
    - none of ``excluded_ids`` (e.g. the pseudo-normal cluster) appears at
      all.
    """
    expected_observed = set(expected_observed)
    roots = desc.index[desc["is_root"]]
    if len(roots) != 1:
        raise AssertionError(f"expected exactly one root, got {list(roots)}")
    root = roots[0]
    if root != germline_id:
        raise AssertionError(f"root is {root!r}, expected {germline_id!r}")
    if desc.loc[germline_id, "observed"]:
        raise AssertionError(f"{germline_id!r} must be latent, but is observed")

    observed_nodes = set(desc.index[desc["observed"]])
    if observed_nodes != expected_observed:
        raise AssertionError(
            f"observed nodes {sorted(observed_nodes)} != expected "
            f"{sorted(expected_observed)}"
        )

    for node in desc.index:
        if node == germline_id or node in expected_observed:
            continue
        if not hidden_pattern.match(node):
            raise AssertionError(
                f"latent node {node!r} matches neither the germline root nor "
                f"the hidden-node pattern {hidden_pattern.pattern!r}"
            )

    excluded = set(excluded_ids) & set(desc.index)
    if excluded:
        raise AssertionError(f"excluded id(s) {sorted(excluded)} appear in the tree")


def assert_channel_order(spectra: pd.DataFrame, cosmic: pd.DataFrame) -> None:
    """Raise if ``spectra``'s columns are not exactly ``cosmic``'s columns,
    in the same order (the positional alignment ``dot(activities,
    signatures)`` depends on)."""
    if list(spectra.columns) != list(cosmic.columns):
        raise AssertionError(
            "spectra columns do not match cosmic_signatures.csv's channel "
            f"order: {list(spectra.columns)[:5]}... vs "
            f"{list(cosmic.columns)[:5]}..."
        )


def assert_nonnegative_integer_counts(df: pd.DataFrame) -> None:
    """Raise if any value is negative or non-integral."""
    values = df.to_numpy(dtype=float)
    if (values < 0).any():
        raise AssertionError("negative counts found")
    if not np.all(np.mod(values, 1) == 0):
        raise AssertionError("non-integer counts found")


def total_snvs_per_cluster(spectra: pd.DataFrame) -> pd.Series:
    """Row sums -- total SNVs binned into each cluster's spectrum."""
    return spectra.sum(axis=1)


def smoke_sample(
    newick_string: str,
    data_matrix: pd.DataFrame,
    fixed_signatures: np.ndarray,
    priors: Optional[dict] = None,
    draws: int = 30,
    tune: int = 30,
    chains: int = 1,
    cores: int = 1,
) -> Tuple[TreeHDP, object]:
    """Build the fixed-signature TreeHDP on ``newick_string`` and run a very
    short sample. Plumbing only: confirms the graph builds and the sampler
    runs without error, not that anything recovers correctly."""
    model = TreeHDP(
        newick_string=newick_string,
        data_matrix=data_matrix,
        priors=priors or DEFAULT_PRIORS,
        fixed_signatures=fixed_signatures,
    )
    trace = model.sample(draws=draws, tune=tune, chains=chains, cores=cores)
    return model, trace


def _format_tree_report(name: str, desc: pd.DataFrame) -> List[str]:
    lines = [f"\n== {name} ==", f"root: {desc.index[desc['is_root']].tolist()}"]
    for node in desc.index:
        row = desc.loc[node]
        kind = "observed" if row["observed"] else "latent"
        lines.append(f"  {node:>10}  parent={row['parent']!s:>10}  {kind}")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tree-dir", type=Path, default=Path("realdata/local_tree_input"))
    p.add_argument(
        "--cosmic-signatures",
        type=Path,
        default=Path("COSMIC_sig/cosmic_signatures.csv"),
    )
    p.add_argument(
        "--signature-subset",
        nargs="+",
        default=["SBS1", "SBS5"],
        help="small signature subset for the smoke sample only (plumbing, "
        "not analysis)",
    )
    p.add_argument(
        "--excluded-cluster-id", default="4", help="the pseudo-normal cluster"
    )
    p.add_argument("--draws", type=int, default=30)
    p.add_argument("--tune", type=int, default=30)
    p.add_argument(
        "--skip-sample", action="store_true", help="skip the smoke-sample step"
    )
    args = p.parse_args()

    cosmic = pd.read_csv(args.cosmic_signatures, index_col=0)
    fixed_signatures = cosmic.loc[args.signature_subset].to_numpy()

    print("## Known pitfall: numeric cluster-ID index")
    raw_spectra = pd.read_csv(args.tree_dir / "spectra.csv", index_col=0)
    print(f"plain pd.read_csv index dtype: {raw_spectra.index.dtype}")
    demo_newick = (args.tree_dir / "snv_tree.nwk").read_text().strip()
    demo_model = TreeHDP(
        newick_string=demo_newick,
        data_matrix=raw_spectra,
        priors=DEFAULT_PRIORS,
        fixed_signatures=fixed_signatures,
    )
    n_obs = len(demo_model.model.observed_RVs)
    print(f"likelihood terms built from the RAW (unfixed) data matrix: {n_obs}")
    if n_obs == 0:
        print(
            "  CONFIRMED: every node is silently dropped from the likelihood, no "
            "error raised -- see read_data_matrix / the module docstring."
        )
    else:
        print("  unexpected: the raw index did not reproduce the pitfall; investigate.")

    spectra = read_data_matrix(args.tree_dir / "spectra.csv")
    print("\n## Channel order and counts")
    assert_channel_order(spectra, cosmic)
    print("spectra.csv columns match cosmic_signatures.csv's channel order: OK")
    assert_nonnegative_integer_counts(spectra)
    print("all counts are non-negative integers: OK")
    print("total SNVs per cluster:")
    for cluster, total in total_snvs_per_cluster(spectra).items():
        print(f"  clone{cluster}: {int(total)}")

    expected_observed = set(spectra.index)
    trees = {
        "snv_tree.nwk": (args.tree_dir / "snv_tree.nwk").read_text().strip(),
        "cna_tree.nwk": (args.tree_dir / "cna_tree.nwk").read_text().strip(),
        "conservative consensus (script-built)": CONSENSUS_NEWICK,
    }

    print("\n## Per-tree loader check")
    for name, newick_string in trees.items():
        model = TreeHDP(
            newick_string=newick_string,
            data_matrix=spectra,
            priors=DEFAULT_PRIORS,
            fixed_signatures=fixed_signatures,
        )
        desc = describe_tree(model)
        for line in _format_tree_report(name, desc):
            print(line)
        assert_tree_contract(
            desc,
            expected_observed=expected_observed,
            excluded_ids={args.excluded_cluster_id},
        )
        print(f"  contract check: OK ({len(desc)} nodes)")

        if args.skip_sample:
            continue
        try:
            smoke_sample(
                newick_string,
                spectra,
                fixed_signatures,
                draws=args.draws,
                tune=args.tune,
            )
            print(f"  smoke sample ({args.draws} draws, 1 chain): OK")
        except Exception as exc:  # noqa: BLE001 -- report, never swallow
            print(f"  smoke sample: FAILED -- {type(exc).__name__}: {exc}")
            raise

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
