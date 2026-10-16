"""eval/coverage.py: Canonical coverage metrics and pass@k estimators.

Pure functions without I/O for assessing candidate heuristics against baseline R(F).

Canonical Definition of "beats R(F)":
A candidate heuristic beats R(F) on a card iff its mean colors across the
split graphs is strictly lower than R(F) by at least eps:
    candidate_mean_colors <= R(F) - eps   (or R(F) - candidate_mean_colors >= eps)
Default eps is 0.05 colors (configurable). Ties do NOT count as wins.
R(F) is defined as min(mean DSATUR, mean RLF) on the identical graph set.
"""

from typing import List, Sequence, Optional, Union, Dict, Any
import math
import numpy as np


def compute_rf(dsatur_mean: float, rlf_mean: float) -> float:
    """Compute R(F) as min(mean DSATUR, mean RLF) on the same graph split."""
    return min(float(dsatur_mean), float(rlf_mean))


def beats_rf(
    candidate_mean_colors: Optional[float],
    rf_mean_colors: float,
    eps: float = 0.05,
) -> bool:
    """Canonical test of whether a candidate heuristic beats R(F).

    Parameters:
      candidate_mean_colors: candidate mean colors on the graph set (None or float).
      rf_mean_colors: reference R(F) = min(mean DSATUR, mean RLF) on the same graphs.
      eps: margin threshold (default: 0.05).

    Returns:
      True iff candidate_mean_colors is finite and <= rf_mean_colors - eps.
      Ties do NOT count as wins.
    """
    if candidate_mean_colors is None:
        return False
    if not math.isfinite(candidate_mean_colors):
        return False
    if not math.isfinite(rf_mean_colors):
        return False
    return (rf_mean_colors - candidate_mean_colors) >= eps


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k (Chen et al. 2021).

    Estimates the probability that at least one of k randomly chosen samples
    without replacement is a winner, given n total generated samples of which
    c are winners:
        pass@k = 1 - binom(n - c, k) / binom(n, k)

    Parameters:
      n: total number of generated samples (n >= 1)
      c: number of winning/correct samples (0 <= c <= n)
      k: budget of samples considered (1 <= k <= n)

    Returns:
      Estimated pass@k probability in [0.0, 1.0].

    Raises:
      ValueError: if k > n, k <= 0, n <= 0, or c not in [0, n].
    """
    if not isinstance(n, (int, np.integer)) or not isinstance(k, (int, np.integer)) or not isinstance(c, (int, np.integer)):
        raise TypeError(f"n, c, k must be integers, got n={type(n)}, c={type(c)}, k={type(k)}")

    n = int(n)
    c = int(c)
    k = int(k)

    if n <= 0:
        raise ValueError(f"n must be a positive integer, got n={n}")
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got k={k}")
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed n ({n})")
    if c < 0 or c > n:
        raise ValueError(f"c ({c}) must satisfy 0 <= c <= n ({n})")

    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    if k == 1:
        return float(c) / float(n)

    # Numerically stable product formulation of:
    # prod_{j=0}^{k-1} (1 - c / (n - j))
    prod = 1.0
    for j in range(k):
        prod *= (1.0 - float(c) / float(n - j))

    return float(np.clip(1.0 - prod, 0.0, 1.0))


def coverage_at_k(
    card_win_counts: Sequence[int],
    n_samples: int,
    k: int,
) -> float:
    """Compute expected coverage@k across cards given win counts per card.

    Coverage@k is the fraction of cards for which at least one winning
    candidate appears in a random sample of size k.

    Parameters:
      card_win_counts: sequence of win counts c_i per card (length C).
      n_samples: total candidates generated per card (n_samples >= k).
      k: target evaluation budget (1 <= k <= n_samples).

    Returns:
      Mean expected coverage@k across cards in [0.0, 1.0].
      Returns 0.0 if card_win_counts is empty.
    """
    if len(card_win_counts) == 0:
        return 0.0

    probs = [pass_at_k(n_samples, c, k) for c in card_win_counts]
    return float(np.clip(np.mean(probs), 0.0, 1.0))


def empirical_coverage_at_k(
    card_candidate_wins: Sequence[Sequence[bool]],
) -> float:
    """Compute empirical coverage@k when exactly k candidate win flags are evaluated.

    Parameters:
      card_candidate_wins: sequence of boolean sequences for each card.
        Each inner sequence has length k (or up to k).

    Returns:
      Fraction of cards with >= 1 winning candidate in [0.0, 1.0].
      Returns 0.0 if card_candidate_wins is empty.
    """
    if len(card_candidate_wins) == 0:
        return 0.0

    has_winner = [1.0 if any(c_wins) else 0.0 for c_wins in card_candidate_wins]
    return float(np.clip(np.mean(has_winner), 0.0, 1.0))


def best_of_k(
    card_candidate_colors: Sequence[Sequence[Optional[float]]],
    rf_colors: Sequence[float],
    k: int,
    failed_savings_fallback: float = 0.0,
) -> Dict[str, Any]:
    """Compute best-of-k quality and savings vs R(F) across cards.

    For each card, selects the candidate with the lowest mean color count
    among the first k candidates.

    Parameters:
      card_candidate_colors: for each card, a sequence of candidate mean color counts
        (None for failed candidates). Length C.
      rf_colors: reference R(F) mean color count for each card. Length C.
      k: sample budget to evaluate (1 <= k).
      failed_savings_fallback: savings assigned to cards where all k candidates failed.

    Returns:
      Dictionary containing:
        - "mean_best_colors": float or None (mean of best valid colors over completed cards)
        - "mean_savings_rf": float (mean savings in colors vs R(F) across all cards)
        - "mean_savings_rf_pct": float (mean percentage savings vs R(F) across all cards)
        - "card_success_rate": float in [0.0, 1.0] (fraction of cards with >= 1 valid candidate)
        - "per_card_best_colors": list of float or None
        - "per_card_savings_rf": list of float
        - "per_card_savings_rf_pct": list of float
    """
    n_cards = len(card_candidate_colors)
    if n_cards == 0:
        return {
            "mean_best_colors": None,
            "mean_savings_rf": 0.0,
            "mean_savings_rf_pct": 0.0,
            "card_success_rate": 0.0,
            "per_card_best_colors": [],
            "per_card_savings_rf": [],
            "per_card_savings_rf_pct": [],
        }

    if len(rf_colors) != n_cards:
        raise ValueError(
            f"Length mismatch: card_candidate_colors ({n_cards}) vs rf_colors ({len(rf_colors)})"
        )
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got k={k}")

    per_card_best_colors: List[Optional[float]] = []
    per_card_savings_rf: List[float] = []
    per_card_savings_rf_pct: List[float] = []
    per_card_success: List[bool] = []

    for i in range(n_cards):
        candidates = card_candidate_colors[i][:k]
        valid_colors = [c for c in candidates if c is not None and math.isfinite(c)]
        rf = float(rf_colors[i])

        if valid_colors:
            best_c = float(min(valid_colors))
            savings = rf - best_c
            savings_pct = (savings / rf * 100.0) if rf > 0.0 else 0.0
            per_card_best_colors.append(best_c)
            per_card_savings_rf.append(savings)
            per_card_savings_rf_pct.append(savings_pct)
            per_card_success.append(True)
        else:
            per_card_best_colors.append(None)
            per_card_savings_rf.append(failed_savings_fallback)
            per_card_savings_rf_pct.append(failed_savings_fallback)
            per_card_success.append(False)

    valid_best = [c for c in per_card_best_colors if c is not None]
    mean_best = float(np.mean(valid_best)) if valid_best else None

    return {
        "mean_best_colors": mean_best,
        "mean_savings_rf": float(np.mean(per_card_savings_rf)),
        "mean_savings_rf_pct": float(np.mean(per_card_savings_rf_pct)),
        "card_success_rate": float(np.mean([1.0 if s else 0.0 for s in per_card_success])),
        "per_card_best_colors": per_card_best_colors,
        "per_card_savings_rf": per_card_savings_rf,
        "per_card_savings_rf_pct": per_card_savings_rf_pct,
    }
