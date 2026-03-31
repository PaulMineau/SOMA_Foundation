"""Loop termination logic for the research loop."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

MARGINAL_GAIN_THRESHOLD = 0.05
MIN_SCORES_FOR_CONVERGENCE = 10


def should_converge(
    score_history: list[float],
    iteration: int,
    max_iterations: int,
) -> bool:
    """Decide whether the research loop should stop."""
    if iteration >= max_iterations:
        logger.info(
            "Converged: reached max iterations (%d)", max_iterations
        )
        return True

    if len(score_history) < MIN_SCORES_FOR_CONVERGENCE:
        return False

    last_10 = score_history[-MIN_SCORES_FOR_CONVERGENCE:]
    marginal_gain = max(last_10) - min(last_10)

    if marginal_gain < MARGINAL_GAIN_THRESHOLD:
        logger.info(
            "Converged: marginal gain %.4f < %.4f over last %d papers",
            marginal_gain,
            MARGINAL_GAIN_THRESHOLD,
            MIN_SCORES_FOR_CONVERGENCE,
        )
        return True

    return False
