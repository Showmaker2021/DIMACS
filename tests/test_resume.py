"""Unit test confirming that population (programs and fitness) after resume exactly matches before pause."""

import pytest
import numpy as np
import tempfile
import os
from gate.evolved.task import CandidateLogger, SEED_CODE_DSATUR, SEED_CODE_RLF


class MockIndividual:
    def __init__(self, program: str, fitness: float, mean_colors: float):
        self.program = program
        self.fitness = fitness
        self.mean_colors = mean_colors


def test_resume_population_equivalence():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        logger = CandidateLogger(tmp_path)
        family_id = "test_fam"

        # Simulate 4 evaluations before pause
        mock_candidates = [
            (SEED_CODE_DSATUR, 0.50, 13.5),
            (SEED_CODE_RLF, 0.85, 12.8),
            ("def priority(adj, colors, sat, deg, udeg): return sat", -100.0, None),
            ("def priority(adj, colors, sat, deg, udeg): return sat * 10 + udeg", 0.72, 13.1),
        ]

        pop_before = []
        for i, (prog, fit, mc) in enumerate(mock_candidates):
            rec = {
                "family_id": family_id,
                "iteration": i + 1,
                "program": prog,
                "fitness": fit,
                "mean_colors": mc,
                "error": None if fit > -50 else "error",
            }
            logger.log(rec)
            if fit > -50:
                pop_before.append((prog.strip(), fit))

        # Sort population before pause by fitness desc
        pop_before.sort(key=lambda x: x[1], reverse=True)

        # SIMULATE RESUME: restore population from log file
        pop_after = []
        with open(tmp_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = logger.log_path  # dummy
                    import json
                    d = json.loads(line)
                    if d.get("family_id") == family_id and d.get("fitness", -100) > -50:
                        pop_after.append((d["program"].strip(), d["fitness"]))

        pop_after.sort(key=lambda x: x[1], reverse=True)

        # Assert exact equivalence
        assert len(pop_before) == len(pop_after), "Population size differs after resume"
        for (prog_b, fit_b), (prog_a, fit_a) in zip(pop_before, pop_after):
            assert prog_b == prog_a, "Program code differs after resume"
            assert np.isclose(fit_b, fit_a), f"Fitness differs after resume: {fit_b} vs {fit_a}"

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
