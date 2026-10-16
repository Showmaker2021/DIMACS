"""Unit and property tests for train/reward.py (GRPO reward engine)."""

import math
import tempfile
from pathlib import Path
import numpy as np
import pytest

from train.reward import (
    VALID_STATUS,
    KNOWN_FAILURE_STATUSES,
    CandidateResult,
    RewardConfig,
    RewardBreakdown,
    compute_advantages,
    compute_group_rewards,
    compute_group_rewards_detailed,
    compute_r_sav,
    compute_r_tb,
    compute_r_time,
    get_reward_bounds,
    kendall_tau_distance,
    pairwise_probe_distance,
)


class TestRewardBasics:
    """Core tests for failure penalties, baseline equivalence, and savings."""

    def test_each_failure_status_yields_minus_one(self):
        """Every failure status must yield exactly -1.0 reward."""
        failure_statuses = list(KNOWN_FAILURE_STATUSES) + ["custom_fail_status", "error"]
        c_base = [10, 15, 20]
        
        candidates = [
            CandidateResult(
                status=status,
                colors_per_graph=[5, 5, 5],  # even with great colors
                runtime_per_graph=[0.001, 0.001, 0.001],
            )
            for status in failure_statuses
        ]
        
        rewards = compute_group_rewards(candidates, c_base_per_graph=c_base)
        assert len(rewards) == len(failure_statuses)
        for r in rewards:
            assert r == -1.0
            assert isinstance(r, float)

        # Also verify in detailed breakdown
        detailed = compute_group_rewards_detailed(candidates, c_base_per_graph=c_base)
        for b, status in zip(detailed, failure_statuses):
            assert b.status == status
            assert b.reward == -1.0
            assert b.r_pre == -1.0
            assert b.r_div == 0.0
            assert not b.is_eligible_diversity

    def test_baseline_identical_candidate_gets_r_sav_zero(self):
        """A candidate matching baseline on all graphs gets r_sav = 0.0."""
        c_base = [12, 14, 16, 18]
        cand = CandidateResult(
            status="ok",
            colors_per_graph=[12, 14, 16, 18],
            runtime_per_graph=[0.01, 0.01, 0.01, 0.01],
        )
        
        # Test compute_r_sav directly
        r_sav = compute_r_sav(cand.colors_per_graph, c_base)
        assert pytest.approx(r_sav, abs=1e-12) == 0.0

        # Test within compute_group_rewards with no runtime penalty (t_dsatur None)
        rewards = compute_group_rewards([cand], c_base_per_graph=c_base)
        assert pytest.approx(rewards[0], abs=1e-12) == 0.0

    def test_better_and_worse_candidates_and_clipping(self):
        """Better candidates get positive r_sav, worse negative, clipped at +/-1.0."""
        c_base = [20, 20]
        c_ref = [10, 10]  # max improvement is 10 colors
        config = RewardConfig()

        # Better candidate: 15 colors -> savings = (20 - 15) / 10 = 0.5
        cand_better = CandidateResult(status="ok", colors_per_graph=[15, 15])
        r_better = compute_r_sav(cand_better.colors_per_graph, c_base, c_ref, config)
        assert pytest.approx(r_better, abs=1e-6) == 0.5

        # Worse candidate: 25 colors -> regression = (20 - 25) / 10 = -0.5
        cand_worse = CandidateResult(status="ok", colors_per_graph=[25, 25])
        r_worse = compute_r_sav(cand_worse.colors_per_graph, c_base, c_ref, config)
        assert pytest.approx(r_worse, abs=1e-6) == -0.5

        # Extreme better: 5 colors -> raw savings = (20 - 5) / 10 = 1.5 -> clipped to 1.0
        cand_extreme_better = CandidateResult(status="ok", colors_per_graph=[5, 5])
        r_extreme_better = compute_r_sav(cand_extreme_better.colors_per_graph, c_base, c_ref, config)
        assert r_extreme_better == 1.0

        # Extreme worse: 35 colors -> raw savings = (20 - 35) / 10 = -1.5 -> clipped to -1.0
        cand_extreme_worse = CandidateResult(status="ok", colors_per_graph=[35, 35])
        r_extreme_worse = compute_r_sav(cand_extreme_worse.colors_per_graph, c_base, c_ref, config)
        assert r_extreme_worse == -1.0

    def test_fallback_when_c_ref_is_none(self):
        """When c_ref is None, denominator defaults to c_base(g) - 1 (denom=1)."""
        c_base = [10, 15]
        # c_ref is None:
        # For graph 0: (10 - 9) / max(1, 10 - 9) = 1 / 1 = 1.0
        # For graph 1: (15 - 15) / max(1, 15 - 14) = 0 / 1 = 0.0
        # Mean = 0.5
        cand = CandidateResult(status="ok", colors_per_graph=[9, 15])
        r_sav = compute_r_sav(cand.colors_per_graph, c_base, c_ref=None)
        assert pytest.approx(r_sav, abs=1e-6) == 0.5


class TestRuntimePenalty:
    """Tests for runtime penalty r_time."""

    def test_runtime_penalty_thresholds(self):
        """Test r_time under budget, at 1.5x budget, 2.0x budget, and 5.0x budget."""
        config = RewardConfig(time_factor=3.0)
        t_dsatur = 0.01  # budget = 3.0 * 0.01 = 0.03

        # Under budget: 0.02s <= 0.03s -> penalty = 0.0
        r_under = compute_r_time([0.02], t_dsatur=t_dsatur, config=config)
        assert r_under == 0.0

        # Exactly at budget: 0.03s -> penalty = 0.0
        r_exact = compute_r_time([0.03], t_dsatur=t_dsatur, config=config)
        assert r_exact == 0.0

        # 1.5x budget: 0.045s -> ratio - 1 = 1.5 - 1 = 0.5
        r_mid = compute_r_time([0.045], t_dsatur=t_dsatur, config=config)
        assert pytest.approx(r_mid, abs=1e-6) == 0.5

        # 2.0x budget: 0.06s -> ratio - 1 = 2.0 - 1 = 1.0
        r_double = compute_r_time([0.06], t_dsatur=t_dsatur, config=config)
        assert pytest.approx(r_double, abs=1e-6) == 1.0

        # 5.0x budget: 0.15s -> clipped to 1.0
        r_huge = compute_r_time([0.15], t_dsatur=t_dsatur, config=config)
        assert r_huge == 1.0

    def test_runtime_penalty_missing_or_zero_dsatur(self):
        """When DSATUR runtime is missing or zero, no penalty is assessed."""
        assert compute_r_time([0.1], t_dsatur=None) == 0.0
        assert compute_r_time([0.1], t_dsatur=0.0) == 0.0
        assert compute_r_time([], t_dsatur=0.01) == 0.0


class TestKendallTauDistance:
    """Tests for Kendall-tau normalized distance calculation."""

    def test_identical_permutations_yield_zero(self):
        """Identical vertex permutations must give 0.0 distance."""
        ord1 = [0, 1, 2, 3, 4]
        ord2 = [0, 1, 2, 3, 4]
        assert kendall_tau_distance(ord1, ord2) == 0.0
        assert kendall_tau_distance(np.array(ord1), np.array(ord2)) == 0.0

    def test_reversed_permutations_yield_one(self):
        """Reversed vertex permutations must give 1.0 distance."""
        ord1 = [0, 1, 2, 3, 4]
        ord2 = [4, 3, 2, 1, 0]
        assert kendall_tau_distance(ord1, ord2) == 1.0
        assert kendall_tau_distance(np.array(ord1), np.array(ord2)) == 1.0

    def test_single_transposition(self):
        """A single adjacent swap out of N*(N-1)/2 pairs gives exact fractional distance."""
        # N=4, total pairs = 4*3/2 = 6 pairs.
        # Swapping adjacent elements [0, 1, 2, 3] -> [0, 2, 1, 3] creates exactly 1 discordant pair.
        ord1 = [0, 1, 2, 3]
        ord2 = [0, 2, 1, 3]
        dist = kendall_tau_distance(ord1, ord2)
        assert pytest.approx(dist, abs=1e-6) == 1.0 / 6.0

    def test_arbitrary_element_permutations(self):
        """Orderings of arbitrary vertex IDs (not 0..N-1) behave consistently."""
        ord1 = [100, 200, 300]
        ord2 = [300, 200, 100]
        assert kendall_tau_distance(ord1, ord2) == 1.0

    def test_edge_cases_empty_and_single(self):
        """None, empty, or single-element inputs yield 0.0 distance safely."""
        assert kendall_tau_distance(None, [1, 2]) == 0.0
        assert kendall_tau_distance([1, 2], None) == 0.0
        assert kendall_tau_distance([], []) == 0.0
        assert kendall_tau_distance([1], [1]) == 0.0

    def test_pairwise_probe_distance(self):
        """Pairwise probe distance computes mean over multiple probe graphs."""
        probe_a = [[0, 1, 2], [0, 1, 2, 3]]
        probe_b = [[2, 1, 0], [0, 1, 2, 3]]  # probe 0 reversed (1.0), probe 1 identical (0.0)
        mean_d = pairwise_probe_distance(probe_a, probe_b)
        assert pytest.approx(mean_d, abs=1e-6) == 0.5


class TestDiversityGating:
    """Tests for behavioral diversity reward gating, median rule, and peer exclusion."""

    def test_diversity_applied_only_strictly_above_median(self):
        """Diversity reward is applied ONLY if r_pre is strictly above the group median."""
        config = RewardConfig(b=0.0, d=0.5)  # diversity active with weight 0.5
        c_base = [20]
        
        # 4 candidates with distinct performance
        # c0: 10 colors -> r_pre = 1.0
        # c1: 15 colors -> r_pre = 0.5
        # c2: 17 colors -> r_pre = 0.3
        # c3: 20 colors -> r_pre = 0.0
        # Group r_pre = [1.0, 0.5, 0.3, 0.0]
        # Median r_pre = (0.5 + 0.3) / 2 = 0.4
        # Candidates above median: c0 (1.0 > 0.4) and c1 (0.5 > 0.4)
        # Candidates at/below median: c2 (0.3 <= 0.4) and c3 (0.0 <= 0.4)
        
        # Give diverse probe orderings:
        # c0 and c1 are completely reversed from c2 and c3
        c0 = CandidateResult(status="ok", colors_per_graph=[0], probe_orderings=[[0, 1, 2]])
        c1 = CandidateResult(status="ok", colors_per_graph=[10], probe_orderings=[[2, 1, 0]])
        c2 = CandidateResult(status="ok", colors_per_graph=[14], probe_orderings=[[0, 1, 2]])
        c3 = CandidateResult(status="ok", colors_per_graph=[20], probe_orderings=[[2, 1, 0]])

        detailed = compute_group_rewards_detailed(
            [c0, c1, c2, c3],
            config=config,
            c_base_per_graph=c_base,
            c_ref_per_graph=[0],
        )
        
        # c0 and c1 must be eligible
        assert detailed[0].is_eligible_diversity
        assert detailed[1].is_eligible_diversity
        assert detailed[0].r_div > 0.0
        assert detailed[1].r_div > 0.0
        assert detailed[0].reward > detailed[0].r_pre
        assert detailed[1].reward > detailed[1].r_pre

        # c2 and c3 must NOT be eligible
        assert not detailed[2].is_eligible_diversity
        assert not detailed[3].is_eligible_diversity
        assert detailed[2].r_div == 0.0
        assert detailed[3].r_div == 0.0
        assert detailed[2].reward == detailed[2].r_pre
        assert detailed[3].reward == detailed[3].r_pre

    def test_diversity_zero_when_d_is_zero(self):
        """When d=0.0 (default), diversity bonus has zero effect on final reward."""
        config = RewardConfig(d=0.0)
        c_base = [20]
        c0 = CandidateResult(status="ok", colors_per_graph=[10], probe_orderings=[[0, 1, 2]])
        c1 = CandidateResult(status="ok", colors_per_graph=[15], probe_orderings=[[2, 1, 0]])

        detailed = compute_group_rewards_detailed([c0, c1], config=config, c_base_per_graph=c_base)
        for b in detailed:
            assert b.reward == b.r_pre

    def test_failed_candidates_excluded_from_diversity_pairs(self):
        """Failed candidates must never be included in pairwise diversity comparisons."""
        config = RewardConfig(b=0.0, d=0.2)
        c_base = [20]

        # c0 is ok and above median
        c0 = CandidateResult(status="ok", colors_per_graph=[10], probe_orderings=[[0, 1, 2]])
        # c1 is failed with a reversed probe ordering
        c1 = CandidateResult(status="ast_fail", probe_orderings=[[2, 1, 0]])
        # c2 is ok with identical probe ordering
        c2 = CandidateResult(status="ok", colors_per_graph=[20], probe_orderings=[[0, 1, 2]])

        # For c0, its only valid peer is c2 (c1 is excluded)
        # Since c0 and c2 have identical probe orderings, tau distance is 0.0!
        detailed = compute_group_rewards_detailed(
            [c0, c1, c2], config=config, c_base_per_graph=c_base
        )
        assert detailed[0].is_eligible_diversity
        # If c1 were included, distance would be (1.0 + 0.0)/2 = 0.5
        # Because c1 is excluded, peer distance is only to c2, so distance is 0.0!
        assert detailed[0].r_div == 0.0
        assert detailed[0].reward == detailed[0].r_pre


class TestGroupAdvantageAndEdgeCases:
    """Tests for advantage normalization, single candidates, and all-fail groups."""

    def test_equal_rewards_group_yields_zero_advantages(self):
        """A group where all candidates have identical rewards produces advantages of exactly 0.0."""
        # Case A: all successful with identical reward
        rewards_ok = [0.4, 0.4, 0.4, 0.4]
        adv_ok = compute_advantages(rewards_ok)
        assert adv_ok.shape == (4,)
        assert np.all(adv_ok == 0.0)
        assert not np.any(np.isnan(adv_ok))
        assert not np.any(np.isinf(adv_ok))

        # Case B: all failed with -1.0
        rewards_fail = [-1.0, -1.0, -1.0]
        adv_fail = compute_advantages(rewards_fail)
        assert adv_fail.shape == (3,)
        assert np.all(adv_fail == 0.0)

    def test_population_standard_deviation_used(self):
        """Verify advantage normalization uses population standard deviation (ddof=0)."""
        rewards = [0.0, 2.0]
        # mean = 1.0
        # population std (ddof=0): sqrt(((0-1)^2 + (2-1)^2)/2) = sqrt(1.0) = 1.0
        # sample std (ddof=1): sqrt(((0-1)^2 + (2-1)^2)/1) = sqrt(2.0) ~ 1.414
        eps = 1e-8
        adv = compute_advantages(rewards, eps_std=eps)
        # With population std: (2.0 - 1.0) / (1.0 + eps) ~ 1.0 / 1.0 = 1.0
        expected_adv_1 = (2.0 - 1.0) / (1.0 + eps)
        assert pytest.approx(adv[1], abs=1e-5) == expected_adv_1
        assert pytest.approx(adv[0], abs=1e-5) == -expected_adv_1

    def test_single_candidate_success_handled_cleanly(self):
        """A group with a single successful candidate executes without exception."""
        cand = CandidateResult(status="ok", colors_per_graph=[10], runtime_per_graph=[0.01])
        rewards = compute_group_rewards([cand], c_base_per_graph=[12])
        assert len(rewards) == 1
        adv = compute_advantages(rewards)
        assert adv.shape == (1,)
        assert adv[0] == 0.0

    def test_all_fail_group_handled_cleanly(self):
        """A group where all candidates fail executes without exception."""
        cands = [
            CandidateResult(status="ast_fail"),
            CandidateResult(status="timeout"),
            CandidateResult(status="memory"),
        ]
        rewards = compute_group_rewards(cands, c_base_per_graph=[10])
        assert rewards == [-1.0, -1.0, -1.0]
        adv = compute_advantages(rewards)
        assert np.all(adv == 0.0)

    def test_single_success_among_failures_handled_cleanly(self):
        """A group with only 1 success and multiple failures executes without error."""
        cands = [
            CandidateResult(status="ok", colors_per_graph=[8]),
            CandidateResult(status="timeout"),
            CandidateResult(status="exception"),
        ]
        config = RewardConfig(d=0.2)
        rewards = compute_group_rewards(cands, config=config, c_base_per_graph=[10])
        assert len(rewards) == 3
        assert rewards[0] > 0.0
        assert rewards[1] == -1.0
        assert rewards[2] == -1.0
        adv = compute_advantages(rewards)
        assert not np.any(np.isnan(adv))
        assert adv[0] > 0.0
        assert adv[1] < 0.0


class TestDeterminismAndConfig:
    """Tests for determinism, serialization, and config handling."""

    def test_determinism(self):
        """Evaluating identical inputs twice produces bitwise identical outputs."""
        c_base = [15, 18, 22]
        cands1 = [
            CandidateResult(status="ok", colors_per_graph=[14, 17, 20], runtime_per_graph=[0.01, 0.02, 0.01]),
            CandidateResult(status="ok", colors_per_graph=[15, 18, 22], runtime_per_graph=[0.02, 0.02, 0.02]),
            CandidateResult(status="timeout"),
        ]
        cands2 = [
            CandidateResult(status="ok", colors_per_graph=[14, 17, 20], runtime_per_graph=[0.01, 0.02, 0.01]),
            CandidateResult(status="ok", colors_per_graph=[15, 18, 22], runtime_per_graph=[0.02, 0.02, 0.02]),
            CandidateResult(status="timeout"),
        ]
        
        r1 = compute_group_rewards(cands1, c_base_per_graph=c_base)
        r2 = compute_group_rewards(cands2, c_base_per_graph=c_base)
        assert r1 == r2

        a1 = compute_advantages(r1)
        a2 = compute_advantages(r2)
        np.testing.assert_array_equal(a1, a2)

    def test_config_from_dict_and_yaml(self):
        """Test RewardConfig loading from dict and YAML file."""
        cfg_dict = {"a": 0.05, "b": 0.2, "d": 0.15, "time_factor": 4.0}
        cfg = RewardConfig.from_dict(cfg_dict)
        assert cfg.a == 0.05
        assert cfg.b == 0.2
        assert cfg.d == 0.15
        assert cfg.time_factor == 4.0

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
            tmp.write("a: 0.1\nb: 0.15\nd: 0.2\ntime_factor: 2.5\n")
            tmp_path = Path(tmp.name)

        try:
            cfg_yaml = RewardConfig.from_yaml(tmp_path)
            assert cfg_yaml.a == 0.1
            assert cfg_yaml.b == 0.15
            assert cfg_yaml.d == 0.2
            assert cfg_yaml.time_factor == 2.5
        finally:
            tmp_path.unlink(missing_ok=True)


class TestMathematicalBoundsProperty:
    """Property-based randomized tests verifying mathematical bounds."""

    def test_reward_values_stay_within_documented_bounds(self):
        """Randomized property test across 500 configurations: all rewards satisfy documented bounds."""
        rng = np.random.RandomState(42)
        
        for _ in range(500):
            # Random config parameters
            a = float(rng.choice([0.0, 0.05, 0.1, 0.2]))
            b = float(rng.uniform(0.0, 0.5))
            d = float(rng.choice([0.0, 0.05, 0.1, 0.2]))
            time_factor = float(rng.uniform(1.0, 5.0))
            config = RewardConfig(a=a, b=b, d=d, time_factor=time_factor)

            lo_bound, hi_bound = get_reward_bounds(config)

            # Generate random group of candidates
            group_size = rng.randint(1, 10)
            n_graphs = rng.randint(1, 5)
            c_base = rng.randint(10, 50, size=n_graphs).tolist()
            c_ref = [max(1, cb - rng.randint(1, 10)) for cb in c_base]
            t_dsatur = rng.uniform(0.005, 0.05, size=n_graphs).tolist()

            candidates = []
            for _ in range(group_size):
                status = "ok" if rng.rand() > 0.2 else rng.choice(list(KNOWN_FAILURE_STATUSES))
                colors = (np.array(c_base) + rng.randint(-5, 15, size=n_graphs)).tolist()
                runtimes = rng.uniform(0.001, 0.2, size=n_graphs).tolist()
                
                # Probe orderings
                n_probe = 5
                p1 = rng.permutation(n_probe).tolist()
                p2 = rng.permutation(n_probe).tolist()
                probe_orderings = [p1, p2]

                cand = CandidateResult(
                    status=status,
                    colors_per_graph=colors,
                    runtime_per_graph=runtimes,
                    probe_orderings=probe_orderings,
                )
                candidates.append(cand)

            rewards = compute_group_rewards(
                candidates,
                config=config,
                c_base_per_graph=c_base,
                c_ref_per_graph=c_ref,
                t_dsatur_per_graph=t_dsatur,
            )

            for r in rewards:
                assert lo_bound - 1e-9 <= r <= hi_bound + 1e-9, (
                    f"Reward {r} out of bounds [{lo_bound}, {hi_bound}] "
                    f"for config a={a}, b={b}, d={d}"
                )
