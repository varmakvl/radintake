"""Generalisation check stays disjoint from the generator templates and reports numbers only."""

from __future__ import annotations

from data_prep.render import INJECTIONS
from radintake.agents.injection import CANONICAL_PHRASES
from radintake.eval.injection_paraphrase import CLEAN, PARAPHRASES, measure, ocr_noise


def test_paraphrases_are_not_templates_or_canonical_phrases():
    low = [t.lower() for t in INJECTIONS]
    for p in PARAPHRASES:
        assert p.lower() not in low
        assert not any(c in p.lower() for c in CANONICAL_PHRASES), p


def test_noise_is_deterministic_and_changes_text():
    assert ocr_noise(PARAPHRASES[0]) == ocr_noise(PARAPHRASES[0]) != PARAPHRASES[0]


def test_measure_is_numbers_only_and_no_false_flags():
    res = measure()
    assert all(isinstance(v, (int, float)) for v in res.values())
    assert res["n_paraphrases"] == len(PARAPHRASES) and res["n_clean"] == len(CLEAN)
    assert res["false_flags_clean"] == 0
