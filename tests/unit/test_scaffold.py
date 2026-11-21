import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]


def test_cli_output_schema_is_valid():
    schema = json.loads((ROOT / "schemas" / "cli_output.v1.json").read_text())
    Draft202012Validator.check_schema(schema)


def test_fixtures_build_without_licensed_data(tmp_path):
    from tests.integration.fixtures import build_corpus

    truth = build_corpus(tmp_path)
    assert (tmp_path / "image_sc_burned.dcm").exists()
    assert len(truth["reports"]) == 4
