from collections.abc import Iterator
from typing import Any

from olmo_eval.core.formatters import CompletionFormatter, PPLFormatter
from olmo_eval.core.metrics import AccuracyMetric, BPBMetric, PassAtKMetric
from olmo_eval.core.scorers import MinervaMathScorer
from olmo_eval.core.types import Instance, LMOutput, LMRequest, SamplingParams
from olmo_eval.data import DataLoader, DataSource
from olmo_eval.evals.extract import MathExtractor
from olmo_eval.evals.tasks.core import Task, TaskConfig, register, register_variant

MATH_SUBSETS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


class MinervaMathTask(Task):
    fewshot_split: str = "train"

    @property
    def instances(self) -> Iterator[Instance]:
        if self._instances_cache is None:
            self._instances_cache = []
            loader = DataLoader()
            source = self._get_source_for_split("test")
            for index, doc in enumerate(loader.load(source)):
                instance = self.process_doc(doc, index)
                if instance is not None:
                    self._instances_cache.append(instance)
        yield from self._instances_cache

    def _get_source_for_split(self, split: str) -> DataSource:
        return self.config.get_data_source(split=split)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        solution_text = doc["solution"]
        extracted_answers = MathExtractor.extract_answer(solution_text)

        primary_answer = extracted_answers[0] if extracted_answers else None
        all_gold_answers = extracted_answers if extracted_answers else []

        return Instance(
            question=doc["problem"],
            gold_answer=primary_answer,
            metadata={
                "id": doc.get("index", index),
                "level": doc.get("level"),
                "type": doc.get("type"),
                "solution_text": solution_text,
                "all_gold_answers": all_gold_answers,
            },
        )

    def format_request(self, instance: Instance) -> LMRequest:
        return self.config.formatter.format(instance, self.get_fewshot())

    def extract_answer(self, output: LMOutput) -> str | None:
        answers = MathExtractor.extract_answer(output.text)
        # Store all extracted answers in metadata for flexible scorers
        output.metadata["all_extracted_answers"] = answers if answers else []
        return answers[0] if answers else None


class Math500Task(MinervaMathTask):
    def _get_source_for_split(self, split: str) -> DataSource:
        # MATH-500 only has test split; use full MATH for train/dev
        if split != "test":
            return DataSource(
                path="EleutherAI/hendrycks_math",
                subset="algebra",  # Use algebra subset for few-shot
                split=split,
            )
        return self.config.get_data_source(split=split)

    def process_doc(self, doc: dict[str, Any], index: int = 0) -> Instance | None:
        solution_text = doc["solution"]
        extracted_answers = MathExtractor.extract_answer(solution_text)

        gold_answer = extracted_answers[0] if extracted_answers else None
        all_gold_answers = extracted_answers if extracted_answers else []

        return Instance(
            question=doc["problem"],
            gold_answer=gold_answer,
            metadata={
                "id": doc.get("index", index),
                "level": doc.get("level"),
                "type": doc.get("type", doc.get("subject")),
                "solution_text": doc.get("solution"),
                "all_gold_answers": all_gold_answers,
            },
        )


def _minerva_math_config(subset: str) -> TaskConfig:
    return TaskConfig(
        name=f"minerva_math_{subset}" if subset else "minerva_math",
        data_source=DataSource(
            path="EleutherAI/hendrycks_math",
            subset=subset,
        ),
        formatter=CompletionFormatter(
            template="Problem: {question}\nSolution: ",
            fewshot_answer_key="solution_text",
        ),
        metrics=(AccuracyMetric(scorer=MinervaMathScorer),),
        num_fewshot=4,
        sampling_params=SamplingParams(
            max_tokens=1024, temperature=0, stop_sequences=["Problem:", "\n\n"]
        ),
    )


def _math500_config() -> TaskConfig:
    return TaskConfig(
        name="math500",
        data_source=DataSource(path="HuggingFaceH4/MATH-500"),
        formatter=CompletionFormatter(
            template="Problem: {question}\nSolution: ",
            fewshot_answer_key="solution_text",
        ),
        metrics=(AccuracyMetric(scorer=MinervaMathScorer),),
        num_fewshot=4,
        sampling_params=SamplingParams(
            max_tokens=1024, temperature=0, stop_sequences=["Problem:", "\n\n"]
        ),
    )


# Metrics for olmes_n4_v2 variant (4 samples, pass@1,2,4; matches oe-eval minerva_math::olmes:n4:v2)
_minerva_pass_at_1 = PassAtKMetric(k=1, scorer=MinervaMathScorer)
_minerva_olmes_n4_v2_metrics = (
    AccuracyMetric(scorer=MinervaMathScorer),
    _minerva_pass_at_1,
    PassAtKMetric(k=2, scorer=MinervaMathScorer),
    PassAtKMetric(k=4, scorer=MinervaMathScorer),
)

for subset in MATH_SUBSETS:
    task_name = f"minerva_math_{subset}"
    class_name = f"MinervaMath_{subset.title().replace('_', '')}"

    task_cls = type(class_name, (MinervaMathTask,), {})
    register(task_name, lambda s=subset: _minerva_math_config(s))(task_cls)


@register("math500", _math500_config)
class Math500(Math500Task):
    pass


for _subset in MATH_SUBSETS:
    _task_name = f"minerva_math_{_subset}"

    register_variant(
        _task_name,
        "olmo3",
        sampling_params=SamplingParams(
            max_tokens=1024, temperature=0.6, top_p=0.6, stop_sequences=["Problem:", "\n\n"]
        ),
    )

    # 4 samples per instance, pass@1/2/4, temperature 0.6 / top_p 0.6 (matches oe-eval olmes:n4:v2)
    # Use olmes_n4_v2 (no colon) so spec minerva_math_X:olmes_n4_v2 parses as one variant
    register_variant(
        _task_name,
        "olmes_n4_v2",
        metrics=_minerva_olmes_n4_v2_metrics,
        primary_metric=_minerva_pass_at_1,
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=0.6,
            top_p=0.6,
            stop_sequences=["Problem:", "\n\n"],
            num_samples=4,
        ),
    )

    register_variant(
        _task_name,
        "bpb",
        formatter=PPLFormatter(),
        metrics=(BPBMetric(),),
        primary_metric=BPBMetric(),
    )

register_variant(
    "math500",
    "bpb",
    formatter=PPLFormatter(),
    metrics=(BPBMetric(),),
    primary_metric=BPBMetric(),
)
