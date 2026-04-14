"""RAEN Scoring for research candidates — Relevance, Actionability, Evidence, Novelty.

Each dimension 0-10. Total /40 normalized to 0-1.
Threshold for corpus inclusion: >= 0.65

Adapted from SOMA AutoResearcher's RAEN scorer for the recommendation corpus.
"""

from __future__ import annotations


def _fuzzy_tag_overlap(candidate_tags: set[str], interest_tags: set[str]) -> int:
    """Count tag matches allowing substring overlap.

    Exact match: "consciousness research" == "consciousness research" -> 1
    Substring: "neuroscience" overlaps with "consciousness research" -> 0
    Word overlap: "AI" in "AI architecture" -> 1
    """
    count = 0
    for ctag in candidate_tags:
        ctag_lower = ctag.lower()
        ctag_words = set(ctag_lower.split())
        for itag in interest_tags:
            itag_lower = itag.lower()
            itag_words = set(itag_lower.split())
            # Exact match, substring match, or word overlap
            if (ctag_lower == itag_lower
                    or ctag_lower in itag_lower
                    or itag_lower in ctag_lower
                    or ctag_words & itag_words):
                count += 1
                break
    return count


def score_candidate(candidate: dict, profile: dict) -> dict:
    """Score a single research candidate against the profile.

    Returns the candidate dict augmented with raen scores.
    """
    scores: dict[str, int] = {}

    # Relevance — does this match interests and current state?
    interest_tags = set(profile["identity"]["interests"])
    candidate_tags = set(candidate.get("tags", []))
    tag_overlap = _fuzzy_tag_overlap(candidate_tags, interest_tags)
    state_match = profile["current_state"]["state"] in candidate.get("best_states", [])
    scores["relevance"] = min(10, tag_overlap * 2 + (4 if state_match else 0))

    # Actionability — can this be done/watched/read now?
    duration = candidate.get("duration_min", 60)
    avoid_states = candidate.get("avoid_states", [])
    current_state = profile["current_state"]["state"]
    blocked = current_state in avoid_states
    scores["actionability"] = 0 if blocked else min(10, max(4, 10 - duration // 30))

    # Evidence — has this type worked before in feedback?
    # Base score of 4 when no feedback exists (benefit of the doubt for new systems).
    # As feedback accumulates, evidence score is earned from actual outcomes.
    worked_types = [w["type"] for w in profile.get("what_worked", [])]
    worked_titles = [w["title"] for w in profile.get("what_worked", [])]
    has_any_feedback = len(worked_types) > 0 or len(profile.get("what_didnt_work", [])) > 0

    if has_any_feedback:
        type_worked = worked_types.count(candidate.get("type", ""))
        title_bonus = 3 if candidate.get("title") in worked_titles else 0
        scores["evidence"] = min(10, type_worked * 3 + title_bonus)
    else:
        # No feedback yet — give a neutral base score
        scores["evidence"] = 4

    # Novelty — is this new to the corpus?
    existing = profile.get("existing_corpus", [])
    is_new = candidate.get("title") not in existing
    scores["novelty"] = 8 if is_new else 2

    total = sum(scores.values())
    normalized = round(total / 40, 3)

    return {
        **candidate,
        "raen": scores,
        "raen_total": normalized,
        "recommended": normalized >= 0.55,
    }


def score_candidates(candidates: list[dict], profile: dict) -> list[dict]:
    """Score and rank a list of candidates."""
    scored = [score_candidate(c, profile) for c in candidates]
    scored.sort(key=lambda x: x["raen_total"], reverse=True)
    return scored
