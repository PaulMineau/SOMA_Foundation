"""Tests for the Layer Specificity Score (LSS) dimension of the RAEN scorer."""

from __future__ import annotations

import pytest

from soma.autoresearcher.extractor import PaperExtract
from soma.autoresearcher.fetcher import Paper
from soma.autoresearcher.scorer import compute_layer_specificity, LAYER_MAP


def _make_paper(title: str = "", abstract: str = "") -> Paper:
    return Paper(
        pmid="12345",
        doi=None,
        title=title,
        abstract=abstract,
        year=2024,
        study_type="RCT",
        citation_count=50,
        industry_funded=False,
        full_text=None,
        source="pubmed",
    )


def _make_extract(
    intervention: str = "",
    outcome: str = "",
    population: str = "",
) -> PaperExtract:
    return PaperExtract(
        intervention=intervention,
        population_description=population,
        effect_size=None,
        effect_direction="positive",
        outcome_measure=outcome,
        safe_for_profile=True,
        actionable=True,
    )


class TestLayerSpecificity:
    """Tests for compute_layer_specificity."""

    def test_proto_self_keywords_score_highest(self) -> None:
        """A paper about HRV and interoception should map to proto_self."""
        paper = _make_paper(
            title="Heart rate variability and interoception in autonomic nervous system regulation",
            abstract="We studied HRV as a marker of homeostasis and body map accuracy.",
        )
        extract = _make_extract(
            intervention="HRV biofeedback training",
            outcome="Autonomic nervous system balance",
        )

        lss, primary_layer, layer_scores = compute_layer_specificity(paper, extract)

        assert primary_layer == "proto_self"
        assert lss > 0.0
        assert layer_scores["proto_self"] > layer_scores["core_consciousness"]

    def test_core_consciousness_keywords(self) -> None:
        """A paper about global workspace and attention should map to core_consciousness."""
        paper = _make_paper(
            title="Global workspace theory and neural correlates of consciousness",
            abstract="Temporal binding and attention as the specious present in working memory.",
        )
        extract = _make_extract(
            intervention="Attention training",
            outcome="Present moment awareness",
        )

        lss, primary_layer, layer_scores = compute_layer_specificity(paper, extract)

        assert primary_layer == "core_consciousness"
        assert lss > 0.0

    def test_extended_consciousness_keywords(self) -> None:
        """A paper about episodic memory consolidation should map to extended_consciousness."""
        paper = _make_paper(
            title="Hippocampal replay and episodic memory consolidation during sleep",
            abstract="Memory reconsolidation in the autobiographical narrative self.",
        )
        extract = _make_extract(
            intervention="Sleep optimization",
            outcome="Long-term memory consolidation",
        )

        lss, primary_layer, layer_scores = compute_layer_specificity(paper, extract)

        assert primary_layer == "extended_consciousness"
        assert lss > 0.0

    def test_relational_self_keywords(self) -> None:
        """A paper about attachment and co-regulation should map to relational_self."""
        paper = _make_paper(
            title="Parent-infant co-regulation and attachment theory",
            abstract="Mirror neuron social cognition and intersubjectivity in theory of mind.",
        )
        extract = _make_extract(
            intervention="Co-regulation therapy",
            outcome="Attachment security",
        )

        lss, primary_layer, layer_scores = compute_layer_specificity(paper, extract)

        assert primary_layer == "relational_self"
        assert lss > 0.0

    def test_no_keywords_returns_zero_lss(self) -> None:
        """A paper with no layer keywords should get LSS=0."""
        paper = _make_paper(
            title="Advances in polymer chemistry",
            abstract="Novel synthesis of polyethylene derivatives.",
        )
        extract = _make_extract(
            intervention="Chemical treatment",
            outcome="Polymer yield",
        )

        lss, primary_layer, layer_scores = compute_layer_specificity(paper, extract)

        assert lss == 0.0

    def test_specificity_rewards_focus(self) -> None:
        """A paper hitting many keywords in ONE layer should score higher LSS
        than a paper hitting keywords across multiple layers equally."""
        # Focused paper — all proto_self keywords
        focused_paper = _make_paper(
            title="HRV interoception homeostasis cardiac afferent autonomic nervous system",
            abstract="Body map and free energy active inference.",
        )
        focused_extract = _make_extract(
            intervention="HRV training",
            outcome="Homeostasis",
        )

        # Diffuse paper — keywords from all layers
        diffuse_paper = _make_paper(
            title="HRV attention episodic memory attachment",
            abstract="Interoception global workspace consolidation co-regulation.",
        )
        diffuse_extract = _make_extract(
            intervention="Multimodal training",
            outcome="General wellbeing",
        )

        focused_lss, _, _ = compute_layer_specificity(focused_paper, focused_extract)
        diffuse_lss, _, _ = compute_layer_specificity(diffuse_paper, diffuse_extract)

        assert focused_lss > diffuse_lss

    def test_layer_scores_are_normalized(self) -> None:
        """Layer scores should be between 0 and 1."""
        paper = _make_paper(
            title="HRV interoception study",
            abstract="Homeostasis and body map in autonomic regulation.",
        )
        extract = _make_extract(intervention="HRV", outcome="ANS balance")

        _, _, layer_scores = compute_layer_specificity(paper, extract)

        for layer_key, score in layer_scores.items():
            assert 0.0 <= score <= 1.0, f"{layer_key} score {score} out of range"


class TestSearchArmsQueries:
    """Verify search arms have queries defined for all layers."""

    def test_all_layers_have_queries(self) -> None:
        from soma.autoresearcher.search_arms import LAYER_QUERIES

        for layer_key in LAYER_MAP:
            assert layer_key in LAYER_QUERIES, f"No queries for {layer_key}"
            assert len(LAYER_QUERIES[layer_key]) >= 2, f"Too few queries for {layer_key}"


class TestBenchmarkStubs:
    """Verify benchmark stubs are runnable."""

    def test_layer_benchmarks_run(self) -> None:
        from soma.benchmarks.layer_benchmarks import run_all_benchmarks

        results = run_all_benchmarks(verbose=False)
        assert len(results) == 5
        for result in results:
            assert result.layer
            assert result.metric_name

    def test_memorial_salience_runs(self) -> None:
        from soma.benchmarks.memorial_salience import run_benchmark

        results = run_benchmark(n_consolidation_cycles=2, verbose=False)
        assert results["n_memories"] == 20
        assert results["pass_criterion"]
