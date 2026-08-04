"""
scripts/run_pipeline.py

Part 11: single entry point that runs the full analysis end-to-end, per the
assignment brief ("The main pipeline entry point should be:
python scripts/run_pipeline.py"). Orchestrates the existing, individually-
tested scripts in order -- it does not duplicate their logic, just calls
each one's main() in sequence, so a fix made in one script is automatically
picked up here too.

Usage:
    python scripts/run_pipeline.py                 # full pipeline, fixed SARIMAX order
    python scripts/run_pipeline.py --grid-search    # also re-run the 147-combination
                                                     # SARIMAX AIC grid search (~35 min)
                                                     # before fitting SARIMAX
    python scripts/run_pipeline.py --skip-chronos   # skip the foundation model step
                                                     # entirely (e.g. on a machine
                                                     # without internet access)

Two steps have real, unavoidable practical constraints on a fresh clone,
handled explicitly rather than silently:

1. SARIMAX order selection (scripts/sarimax_grid_search.py) takes ~35
   minutes for the full 147-combination AIC grid search. By default this
   pipeline SKIPS re-running it and uses the order already selected by that
   search (order=(1,1,6), seasonal_order=(1,1,1,24) -- see report.md,
   Section 6) baked into scripts/run_sarimax.py. Pass --grid-search to
   redo the search from scratch.

2. The foundation model (scripts/run_chronos.py) downloads pretrained
   weights from Hugging Face Hub the first time it runs, which requires
   outbound internet access to huggingface.co. On a machine with normal
   internet access this just works; on a network-restricted machine (e.g.
   a sandboxed CI environment) it will fail. This pipeline attempts it and
   catches the failure with a clear message rather than crashing the whole
   run -- pass --skip-chronos to skip it outright and save the attempt.
"""

import argparse
import importlib.util
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

SCRIPTS_DIR = Path(__file__).resolve().parent


def load_script_module(script_name: str):
    """Import scripts/<script_name> as a module and return it, without
    executing its `if __name__ == "__main__"` block (that only runs when
    the file is executed directly, not when imported like this)."""
    path = SCRIPTS_DIR / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_step(step_name: str, script_name: str):
    print(f"\n{'=' * 70}\n{step_name}\n{'=' * 70}")
    t0 = time.time()
    module = load_script_module(script_name)
    module.main()
    print(f"\n[{step_name} done in {time.time() - t0:.1f}s]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-search", action="store_true",
        help="Re-run the full 147-combination SARIMAX AIC grid search (~35 min) "
             "before fitting SARIMAX. By default this is skipped and the "
             "already-selected order (1,1,6) is used directly.",
    )
    parser.add_argument(
        "--skip-chronos", action="store_true",
        help="Skip the foundation model step entirely (useful on a machine "
             "without internet access to Hugging Face Hub).",
    )
    args = parser.parse_args()

    pipeline_start = time.time()

    # 1. Data pipeline (Part 1): download/cache raw data, clean, resample to hourly.
    print(f"\n{'=' * 70}\nPart 1a: Data pipeline\n{'=' * 70}")
    t0 = time.time()
    sys.path.insert(0, str(SCRIPTS_DIR.parent / "src"))
    from appliance_energy.data import load_appliance_data
    load_appliance_data()
    print(f"\n[Part 1a: Data pipeline done in {time.time() - t0:.1f}s]")

    # 1b. EDA + stationarity (Part 1)
    run_step("Part 1b: EDA and stationarity tests", "eda_and_stationarity.py")

    if args.grid_search:
        print(f"\n{'=' * 70}\nPart 4a: SARIMAX AIC grid search (147 combinations, ~35 min)\n{'=' * 70}")
        t0 = time.time()
        # sarimax_grid_search.py's own argparse would otherwise try (and fail)
        # to parse run_pipeline.py's --grid-search/--skip-chronos flags, since
        # parser.parse_args() reads sys.argv by default -- isolate it here.
        saved_argv = sys.argv
        sys.argv = ["sarimax_grid_search.py"]
        try:
            module = load_script_module("sarimax_grid_search.py")
            module.main()
        finally:
            sys.argv = saved_argv
        print(f"\n[Part 4a: SARIMAX grid search done in {time.time() - t0:.1f}s]")
    else:
        print(
            "\n[Skipping SARIMAX grid search -- using already-selected order "
            "(1,1,6). Pass --grid-search to redo the full 147-combination search.]"
        )

    # 2. Benchmarks (Part 3)
    run_step("Part 3: Benchmark models", "run_benchmarks.py")

    # 3. SARIMAX (Part 4)
    run_step("Part 4b: SARIMAX (target-only)", "run_sarimax.py")
    run_step("Part 4c: SARIMAX (with exogenous variables)", "run_sarimax_exog.py")

    # 4. Feature-based model (Parts 5-6)
    run_step("Parts 5-6: Feature-based model (XGBoost)", "run_feature_model.py")

    # 5. Foundation model (Part 7) -- best-effort, needs internet access.
    if args.skip_chronos:
        print("\n[Skipping foundation model step (--skip-chronos passed).]")
    else:
        print(f"\n{'=' * 70}\nPart 7: Foundation model (Chronos)\n{'=' * 70}")
        try:
            t0 = time.time()
            module = load_script_module("run_chronos.py")
            module.main()
            print(f"\n[Foundation model done in {time.time() - t0:.1f}s]")
        except Exception as exc:  # noqa: BLE001
            print(
                f"\n[Foundation model step failed: {exc}\n"
                "This step needs outbound internet access to huggingface.co to "
                "download pretrained weights. If this machine has no internet "
                "access, that's expected -- rerun elsewhere with `python "
                "scripts/run_chronos.py`, or pass --skip-chronos next time to "
                "skip this attempt. Continuing with the remaining models' "
                "results already on disk from a previous run, if any.]"
            )

    # 6. Full comparison (Part 8) -- requires all model forecasts already merged
    # into outputs/forecasts/all_forecasts.csv, including foundation_model if
    # the step above succeeded (or from a previous run).
    try:
        run_step("Part 8: Full model comparison", "run_full_comparison.py")
    except Exception as exc:  # noqa: BLE001
        print(
            f"\n[Full comparison step failed: {exc}\n"
            "This usually means outputs/forecasts/all_forecasts.csv is missing "
            "the foundation_model column -- see the note above if the Chronos "
            "step was skipped or failed.]"
        )

    total = time.time() - pipeline_start
    print(f"\n{'=' * 70}\nPipeline complete in {total / 60:.1f} minutes.\n{'=' * 70}")
    print(
        "\nOutputs:\n"
        "  outputs/figures/    -- all plots\n"
        "  outputs/forecasts/all_forecasts.csv  -- every model's forecast\n"
        "  outputs/metrics/model_comparison.csv -- final metric comparison\n"
        "  reports/report.docx, reports/report.pdf -- the written report\n"
    )


if __name__ == "__main__":
    main()
