"""Metrics computed from results/runs/*.jsonl joined with dataset.csv. Owner: E.

Definitions:
  answer_correctness      correct supported answers / all questions
  selective_accuracy      correct answers / answered questions
  coverage                answered / all
  abstention_recall       correctly refused insufficient-evidence Qs / all insufficient-evidence Qs
  over_refusal_rate       refused sufficient-evidence Qs / all sufficient-evidence Qs
  unsupported_answer_rate answers not substantiated by retrieved evidence / generated answers
  latency                 mean and p95 latency_ms

TODO(E): implement, stratified by scenario (A/B/C) and method.
"""
