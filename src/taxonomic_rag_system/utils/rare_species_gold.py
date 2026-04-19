"""
Gold-label helpers for the imageomics/rare-species dataset.

This module intentionally avoids decoding images. It only loads the taxonomy
columns needed for evaluation and aligns them by `rarespecies_id` (RSID).
"""

from __future__ import annotations

from typing import Any


RANKS: list[str] = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]


def build_rsid_to_true_class(
    rsids: set[str],
    *,
    only_arthropoda: bool = True,
) -> dict[str, dict[str, str]]:
    """Build a mapping `RSID` -> gold taxonomy dict (no RSID field).

    Parameters
    ----------
    rsids
        Set of RSIDs to fetch labels for.
    only_arthropoda
        Match the existing evaluator behaviour by filtering to `phylum == Arthropoda`.

    Returns
    -------
    dict
        Mapping rsid (str) -> dict of taxonomic ranks (Kingdom..Species).
        Missing RSIDs will simply be absent from the returned mapping.
    """

    if not rsids:
        return {}

    # Lazy import so base package can be installed without datasets.
    from datasets import load_dataset  # type: ignore[import-not-found]

    # Load only the taxonomy columns to avoid image decoding/IO.
    keep_cols = [
        "rarespecies_id",
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "sciName",
    ]

    ds = load_dataset("imageomics/rare-species", split="train")
    try:
        ds = ds.select_columns(keep_cols)
    except Exception:
        # Older datasets versions may not have select_columns; fall back.
        remove_cols = [c for c in ds.column_names if c not in keep_cols]
        if remove_cols:
            ds = ds.remove_columns(remove_cols)

    out: dict[str, dict[str, str]] = {}

    # Iteration is fast enough for this dataset size; avoids filter/map overhead and
    # keeps the code simple and deterministic.
    for row in ds:
        try:
            rsid_val = row.get("rarespecies_id")
        except AttributeError:
            continue
        if rsid_val is None:
            continue
        rsid = str(rsid_val)
        if rsid not in rsids:
            continue
        if only_arthropoda and row.get("phylum") != "Arthropoda":
            continue

        def _s(x: Any) -> str:
            return "" if x is None else str(x)

        out[rsid] = {
            "Kingdom": _s(row.get("kingdom")),
            "Phylum": _s(row.get("phylum")),
            "Class": _s(row.get("class")),
            "Order": _s(row.get("order")),
            "Family": _s(row.get("family")),
            "Genus": _s(row.get("genus")),
            "Species": _s(row.get("sciName")),
        }

    return out

