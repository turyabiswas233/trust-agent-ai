"""Week 02 pipeline integration: extraction, evaluation, reporting, and contract testing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from week02_evaluate import run_evaluation
from week02_extract import run_extraction_pipeline


def run_contract_tests() -> bool:
    """Run schema contract tests using pytest."""
    print("\n" + "=" * 60)
    print("  Running Schema Contract Tests (tests/test_schema_contract.py)")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_schema_contract.py", "-v"],
        capture_output=False,
    )
    if result.returncode == 0:
        print("\n[SUCCESS] All schema contract tests passed (additive accepted, breaking failed).\n")
        return True
    else:
        print(f"\n[FAILURE] Contract tests exited with code {result.returncode}.\n")
        return False


def run_week02_assignment(
    model_name: str | None = None,
    runs: int = 3,
    dry_run: bool = False,
    evaluate_only: bool = False,
    test_contract: bool = False,
    output_dir: Path = Path("output/week02"),
) -> dict[str, Any]:
    """Execute the complete Week 02 assignment pipeline.

    1. (Optional) Run schema contract tests.
    2. Run Prompt-Only (Method A) & Grammar-Constrained (Method B) extractions.
    3. Evaluate extractions against answer key and schema.
    4. Generate 5 matplotlib charts and markdown/CSV reports.
    """
    print("\n" + "=" * 70)
    print("  Week 02 Assignment: Structured Extraction & Grammar Constraints")
    print("  Corpus: 50 Domain Notices | Ledger: Class 1 Wrapper ($0.15/$0.60 per 1M)")
    print("=" * 70)

    if test_contract:
        passed = run_contract_tests()
        if not passed and not evaluate_only and not dry_run:
            print("Warning: Contract tests had failures. Continuing with pipeline...")

    extractions_path = output_dir / "extractions_raw.jsonl"

    if not evaluate_only:
        print(f"\n[Phase 1] Running Extractions (runs={runs}, dry_run={dry_run})...")
        run_extraction_pipeline(
            model_name=model_name,
            runs=runs,
            dry_run=dry_run,
            output_dir=output_dir,
        )
    else:
        if not extractions_path.exists():
            print(f"\n[ERROR] Cannot evaluate-only: {extractions_path} does not exist.")
            print("Please run extractions first (e.g. 'python main.py --week2' or '--week2 --dry-run').\n")
            return {}
        print(f"\n[Phase 1] Skipped extractions (--evaluate-only). Using {extractions_path}")

    print("\n[Phase 2] Evaluating Extractions & Generating Charts...")
    summary = run_evaluation(
        extractions_path=extractions_path,
        output_dir=output_dir,
    )

    print("\n" + "=" * 70)
    print("  WEEK 02 EVALUATION COMPLETE")
    print("=" * 70)
    print(f"  Report:   {output_dir / 'WEEK02_REPORT.md'}")
    print(f"  CSV:      {output_dir / 'comparison_report.csv'}")
    print(f"  Metrics:  {output_dir / 'metrics_summary.json'}")
    print(f"  Charts:   {output_dir / 'charts'} (5 charts generated)")
    print("=" * 70 + "\n")

    return summary
