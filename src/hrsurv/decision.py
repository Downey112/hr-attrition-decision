"""Retention decision rule. All money is in units of one annual salary, so salary cancels out.

  k  cost of losing the employee (replacement, ramp-up, lost output), multiple of salary
  c  cost of the retention intervention, multiple of salary
  e  effectiveness: share of would-be leavers the intervention keeps

Intervening on someone with leaving probability p saves p*e*k and costs c, so it pays when
p > c / (e*k). That threshold needs calibrated probabilities, not just a good ranking.
"""
import numpy as np


def threshold(k: float, c: float, e: float) -> float:
    return c / (e * k)


def break_even_effectiveness(p, k: float, c: float):
    """Smallest effectiveness at which intervening on each employee pays for itself
    (>1 = never; p = 0 gives inf, a missing p stays NaN)."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore"):
        return c / (p * k)


def realised_cost(target, left, k: float, c: float, e: float, w=None) -> float:
    """Mean cost per employee of a targeting policy, given who actually left.

    Targeted: pay c, and lose a leaver with probability (1-e). Not targeted: lose every leaver.
    `w` are optional inverse-probability-of-censoring weights.
    """
    target = np.asarray(target, bool)
    left = np.asarray(left, float)
    w = np.ones_like(left) if w is None else np.asarray(w, float)
    cost = np.where(target, c + (1 - e) * k * left, k * left)
    return float(np.sum(w * cost) / np.sum(w))
