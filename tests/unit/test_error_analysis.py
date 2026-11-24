"""Failure categories used by the error-analysis tool."""

import pytest

from radintake.eval.error_analysis import categorise

GT = {"indication": "Cough and fever.", "findings": "The lungs are clear. No effusion.",
      "impression": "No acute disease.", "comparison": "None."}
PRINTED = "cough and fever. the lungs are clear. no effusion. no acute disease. none. technologist note: ok."


@pytest.mark.parametrize("field, pred, hw, expected", [
    ("findings", "The lungs are clear. No effusion.", False, None),
    ("findings", "", False, "omission"),
    ("findings", "Large right pneumothorax with tension.", False, "hallucination"),
    ("indication", "Cough and fevr.", True, "handwriting"),
    ("findings", "No acute disease.", False, "wrong field"),
    ("impression", "No acute disease. Technologist note: ok.", False, "layout: bleed"),
    ("findings", "The lungs are clear.", False, "layout: truncated"),
    ("findings", "The lungs are dear. No effusion.", False, "OCR misread"),
])
def test_categorise(field, pred, hw, expected):
    assert categorise(field, pred, GT[field], GT, PRINTED, hw) == expected
