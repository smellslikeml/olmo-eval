"""Tests for the IFBench task and IFEval scoring stack."""

from __future__ import annotations

import unittest
from typing import Any

import pytest

from olmo_eval.common.metrics import (
    IFEvalInstLooseAccuracy,
    IFEvalInstStrictAccuracy,
    IFEvalPromptLooseAccuracy,
    IFEvalPromptStrictAccuracy,
)
from olmo_eval.common.scorers import IFEvalScorer
from olmo_eval.common.types import Instance, LMOutput, LMRequest, RequestType, Response
from olmo_eval.evals.tasks.common import get_task

# Skip the whole module if upstream IFBench isn't installed.
pytest.importorskip("instructions_registry")


def _make_instance(
    prompt: str,
    instruction_id_list: list[str],
    kwargs_list: list[dict[str, Any]],
) -> Instance:
    return Instance(
        question=prompt,
        gold_answer=None,
        metadata={
            "id": "test",
            "key": "test",
            "prompt": prompt,
            "instruction_id_list": instruction_id_list,
            "kwargs": kwargs_list,
        },
    )


def _make_response(instance: Instance, response_text: str) -> Response:
    return Response(
        instance=instance,
        request=LMRequest(request_type=RequestType.COMPLETION, prompt=instance.question),
        outputs=[LMOutput(text=response_text)],
    )


class TestIFBenchTask(unittest.TestCase):
    def test_registered(self) -> None:
        task = get_task("ifbench")
        self.assertIsNotNone(task)
        metric_names = {m.name for m in task.config.metrics}
        self.assertEqual(
            metric_names,
            {
                "prompt_level_strict_acc",
                "prompt_level_loose_acc",
                "inst_level_strict_acc",
                "inst_level_loose_acc",
            },
        )

    def test_process_doc_strips_none_kwargs(self) -> None:
        task = get_task("ifbench")
        doc = {
            "key": 0,
            "prompt": "hi",
            "instruction_id_list": ["count:numbers"],
            "kwargs": [{"N": 2, "keyword": None, "frequency": None}],
        }
        instance = task.process_doc(doc, index=0)
        self.assertIsNotNone(instance)
        assert instance is not None
        self.assertEqual(instance.metadata["instruction_id_list"], ["count:numbers"])
        self.assertEqual(instance.metadata["kwargs"], [{"N": 2}])
        self.assertEqual(instance.metadata["key"], 0)


class TestIFEvalScorer(unittest.TestCase):
    """Use ``count:numbers`` (must include exactly N digits) — a deterministic verifier."""

    INSTRUCTION_ID = "count:numbers"
    KWARGS: dict[str, Any] = {"N": 2}

    def _score(self, response_text: str) -> dict[str, list[bool]]:
        instance = _make_instance(
            "Include exactly 2 numbers.", [self.INSTRUCTION_ID], [self.KWARGS]
        )
        output = LMOutput(text=response_text)
        IFEvalScorer().score(instance, output)
        return output.metadata["ifeval"]

    def test_satisfied(self) -> None:
        result = self._score("I have 3 apples and 4 oranges.")
        self.assertEqual(result["strict"], [True])
        self.assertEqual(result["loose"], [True])

    def test_violated_too_few(self) -> None:
        result = self._score("I have apples and oranges.")
        self.assertEqual(result["strict"], [False])
        self.assertEqual(result["loose"], [False])

    def test_violated_too_many(self) -> None:
        result = self._score("I have 1 apple, 2 oranges, and 3 grapes.")
        self.assertEqual(result["strict"], [False])
        self.assertEqual(result["loose"], [False])

    def test_loose_strips_leading_line(self) -> None:
        # Strict fails: leading "Sure!" line adds no number, body has 2.
        # But loose strips first line, leaving exactly 2 — passes.
        result = self._score(
            "Sure! Here's a response with 5 trailing extras at the end.\n"
            "I have 3 apples and 4 oranges."
        )
        self.assertEqual(result["strict"], [False])
        self.assertEqual(result["loose"], [True])


class TestIFEvalMetrics(unittest.TestCase):
    INSTRUCTION_ID = "count:numbers"
    KWARGS: dict[str, Any] = {"N": 2}

    def _scored_response(self, response_text: str) -> Response:
        instance = _make_instance("count.", [self.INSTRUCTION_ID], [self.KWARGS])
        response = _make_response(instance, response_text)
        IFEvalScorer().score(instance, response.outputs[0])
        return response

    def test_aggregation(self) -> None:
        responses = [
            self._scored_response("I have 3 apples and 4 oranges."),  # passes
            self._scored_response("Just words, no digits."),  # fails
        ]
        self.assertAlmostEqual(IFEvalPromptStrictAccuracy().compute(responses), 0.5)
        self.assertAlmostEqual(IFEvalPromptLooseAccuracy().compute(responses), 0.5)
        self.assertAlmostEqual(IFEvalInstStrictAccuracy().compute(responses), 0.5)
        self.assertAlmostEqual(IFEvalInstLooseAccuracy().compute(responses), 0.5)

    def test_multi_instruction_prompt_requires_all(self) -> None:
        instance = _make_instance(
            "Two numbers and an emoji-terminated sentence.",
            ["count:numbers", "format:emoji"],
            [{"N": 2}, {}],
        )
        # Has 2 numbers (passes count) but no emoji at end of sentence (fails emoji).
        response = _make_response(instance, "I have 3 apples and 4 oranges.")
        IFEvalScorer().score(instance, response.outputs[0])
        prompt_acc = IFEvalPromptStrictAccuracy().compute([response])
        inst_acc = IFEvalInstStrictAccuracy().compute([response])
        self.assertEqual(prompt_acc, 0.0)  # not all instructions satisfied
        self.assertEqual(inst_acc, 0.5)  # 1 of 2 satisfied


if __name__ == "__main__":
    unittest.main()
