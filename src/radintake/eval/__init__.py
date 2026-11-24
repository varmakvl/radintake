"""Evaluation harness: scoring against ground truth, runs, offline re-scoring and the CI gate.

* :mod:`radintake.eval.metrics` - pure scoring + aggregation (no I/O besides reading files given).
* :mod:`radintake.eval.run` - run the pipeline on a split and write a run record.
* :mod:`radintake.eval.rescore` - recompute every metric from a run record, models forbidden.
* :mod:`radintake.eval.gate` - CI regression gate on a licence-free synthetic subset.
* :mod:`radintake.eval.versions` - prompt / schema / model-registry lock.
"""
