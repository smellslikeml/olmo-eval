"""Biology evaluation suite."""

from olmo_eval.evals.suites.registry import make_suite

# =============================================================================
# LAB-Bench Suite
# =============================================================================

LAB_BENCH = make_suite(
    "lab_bench",
    (
        "lab_bench_litqa2",
        "lab_bench_dbqa",
        "lab_bench_seqqa",
        "lab_bench_protocolqa",
        "lab_bench_suppqa",
        "lab_bench_cloning_scenarios",
    ),
    description="LAB-Bench biology research benchmark (futurehouse/lab-bench)",
)
