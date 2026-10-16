"""Tests comparing evaluator DSATUR with dataset provisional labels."""

import pytest
import numpy as np
from gate.dataset import families, load_family
from evaluator.coloring import dsatur
from evaluator.verify import verify, count_colors


def test_dsatur_matches_dataset_provisional_labels():
    """Verify evaluator DSATUR on test graphs closely tracks provisional labels."""
    fams = families()

    # Select representative families across structure types
    seen_types = set()
    sample_fams = []
    for f in fams:
        st = f["structure_type"]
        if st not in seen_types and f["labels"].get("dsatur_test_mean_provisional") is not None:
            seen_types.add(st)
            sample_fams.append(f)
        if len(sample_fams) >= 8:
            break

    for f in sample_fams:
        fid = f["family_id"]
        prov_mean = f["labels"]["dsatur_test_mean_provisional"]
        graphs = load_family(fid, "test")

        color_counts = []
        for seed, adj in graphs:
            col = dsatur(adj)
            assert verify(adj, col)
            color_counts.append(count_colors(col))

        eval_mean = float(np.mean(color_counts))
        # Small difference due to tie-breaking is expected (< 0.5 colors)
        diff = abs(eval_mean - prov_mean)
        assert diff < 0.5, (
            f"Family {fid}: DSATUR test mean ({eval_mean:.2f}) differs from "
            f"provisional label ({prov_mean:.2f}) by {diff:.2f} colors"
        )
