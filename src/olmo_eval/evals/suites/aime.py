from olmo_eval.evals.suites.registry import make_suite

make_suite(
    "aime_pass_at_32",
    (
        "aime_2024:pass_at_32",
        "aime_2025:pass_at_32",
    ),
    description="AIME 2024+2025 pass@32 evaluation",
)
