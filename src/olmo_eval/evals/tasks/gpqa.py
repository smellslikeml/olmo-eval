"""GPQA (Graduate-Level Google-Proof Q&A) evaluation tasks.

GPQA (Idavidrein/gpqa) is a PhD-level multiple-choice benchmark spanning
biology, physics, and chemistry. Questions are designed to be "Google-proof" —
difficult enough that non-experts cannot answer them even with web access.

Subsets:
    gpqa_diamond    Highest quality — both expert annotators correct (198 questions)
    gpqa_main       Filtered for quality (448 questions)
    gpqa_extended   Full question set (546 questions)

Each task supports :mc (logprob-based) and :bpb (bits-per-byte) variants.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterator, Sequence
from typing import Any

from olmo_eval.common.formatters import MCQAChatFormatter, MultipleChoiceFormatter, PPLFormatter
from olmo_eval.common.metrics import AccuracyMetric, BPBMetric
from olmo_eval.common.scorers import MultipleChoiceScorer
from olmo_eval.common.types import (
    Instance,
    LMOutput,
    LMRequest,
    RequestType,
    Response,
    SamplingParams,
    Split,
)
from olmo_eval.data import DataSource
from olmo_eval.evals.tasks.common import Task, register, register_variant

# Regex for "ANSWER: X" pattern (case-insensitive)
_ANSWER_PATTERN = re.compile(r"ANSWER\s*:\s*([A-Z])", re.IGNORECASE)

# Fallback: letter in parentheses like (A), (B), etc.
_PAREN_PATTERN = re.compile(r"\(([A-D])\)", re.IGNORECASE)

# Last resort: standalone letter A-D (word boundary on both sides)
_STANDALONE_PATTERN = re.compile(r"\b([A-D])\b")

# Clean up citation-style brackets from question text
_TITLE_PATTERN = re.compile(r"\[title\]")
_BRACKET_PATTERN = re.compile(r"\[.*?\]")


def _clean_text(text: str) -> str:
    """Preprocess GPQA text: strip citation brackets and collapse whitespace."""
    text = _TITLE_PATTERN.sub(".", text)
    text = _BRACKET_PATTERN.sub("", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()


_SYSTEM_PROMPT = """\
You are an expert scientist. Answer the following multiple choice \
question by reasoning through the options and selecting the best answer.

End your response with "ANSWER: X" where X is the letter of your chosen answer."""

_DEFAULT_ACCURACY = AccuracyMetric(scorer=MultipleChoiceScorer)
_DEFAULT_METRICS = (_DEFAULT_ACCURACY,)
_DEFAULT_SAMPLING = SamplingParams(temperature=0.0, max_tokens=1024)


class GPQATask(Task):
    """Base class for GPQA subtasks."""

    # GPQA only publishes a "train" split on HuggingFace
    split = Split.TRAIN

    def __init_subclass__(cls, subset: str | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if subset is None:
            return
        name = subset  # already in snake_case: gpqa_diamond, gpqa_main, gpqa_extended
        cls.data_source = DataSource(path="Idavidrein/gpqa", subset=subset)
        cls.formatter = MCQAChatFormatter(system_prompt=_SYSTEM_PROMPT)
        cls.metrics = _DEFAULT_METRICS
        cls.primary_metric = _DEFAULT_ACCURACY
        cls.sampling_params = _DEFAULT_SAMPLING
        register(name)(cls)
        register_variant(name, "mc", formatter=MultipleChoiceFormatter(), metrics=_DEFAULT_METRICS)
        register_variant(name, "bpb", formatter=PPLFormatter(), metrics=(BPBMetric(),))

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = []
            from olmo_eval.data import DataLoader

            loader = DataLoader()
            source = self.config.get_data_source()
            for idx, doc in enumerate(loader.load(source)):
                instance = self.process_doc(doc, idx)
                if instance is not None:
                    self._instances_cache.append(instance)
        yield from self._instances_cache

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        """Convert a GPQA document to an Instance with shuffled choices."""
        question = doc.get("Question", "")
        correct = doc.get("Correct Answer", "")
        if not question or not correct:
            return None

        question = _clean_text(question)
        correct = _clean_text(correct)

        # Build 4 choices
        choices = [correct]
        for key in ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"):
            val = doc.get(key, "")
            if val:
                choices.append(_clean_text(val))

        # Deterministic per-question shuffle
        rng = random.Random(f"{self.config.seed}:{index}")
        rng.shuffle(choices)

        gold_idx = choices.index(correct)
        gold_letter = chr(ord("A") + gold_idx)

        metadata: dict[str, Any] = {
            "index": index,
            "gold_idx": gold_idx,
            "gold_text": correct,
        }
        explanation = doc.get("Explanation", "")
        if explanation:
            metadata["explanation"] = explanation

        return Instance(
            question=question,
            gold_answer=gold_letter,
            choices=tuple(choices),
            metadata=metadata,
        )

    @property
    def request_type(self) -> RequestType:
        if self.config.formatter is not None:
            return self.config.formatter.request_type
        return RequestType.CHAT

    def format_request(self, instance: Instance) -> LMRequest:
        if self.config.formatter is not None:
            return self.config.formatter.format(instance, self.get_fewshot())
        return LMRequest(
            request_type=RequestType.CHAT,
            messages=({"role": "user", "content": instance.question},),
        )

    def _extract_answers(self, responses: Sequence[Response]) -> None:
        """Extract answers, with argmax for logprob-based MC scoring."""
        for response in responses:
            if (
                response.request.request_type == RequestType.LOGLIKELIHOOD
                and len(response.outputs) > 1
            ):
                # MC logprob mode: pick the continuation with highest logprob
                best_idx = 0
                best_logprob = float("-inf")
                for i, output in enumerate(response.outputs):
                    logprob = (
                        output.metadata.get("total_logprob", float("-inf"))
                        if output.metadata
                        else float("-inf")
                    )
                    if logprob > best_logprob:
                        best_logprob = logprob
                        best_idx = i
                letter = chr(ord("A") + best_idx)
                for output in response.outputs:
                    output.extracted_answer = letter
            else:
                # Chat mode: parse answer from generated text
                for output in response.outputs:
                    output.extracted_answer = self.extract_answer(output)

    def extract_answer(self, output: LMOutput) -> str | None:
        """Extract answer letter from model output.

        Tries in order: "ANSWER: X", "(X)", standalone letter.
        """
        text = output.text
        if not text or not text.strip():
            return None

        # Primary: ANSWER: X
        matches = list(_ANSWER_PATTERN.finditer(text))
        if matches:
            return matches[-1].group(1).upper()

        # Fallback: (A)/(B)/(C)/(D)
        matches = list(_PAREN_PATTERN.finditer(text))
        if matches:
            return matches[-1].group(1).upper()

        # Last resort: standalone A-D letter
        matches = list(_STANDALONE_PATTERN.finditer(text))
        if matches:
            return matches[-1].group(1).upper()

        return None


# =============================================================================
# Task Registrations
# =============================================================================


class GPQADiamond(GPQATask, subset="gpqa_diamond"): ...


class GPQAMain(GPQATask, subset="gpqa_main"): ...


class GPQAExtended(GPQATask, subset="gpqa_extended"): ...
