#!/usr/bin/env python3
"""Golden dataset evaluation runner.

Runs the full evaluation pipeline against the curated golden dataset
and generates a report with retrieval + generation metrics.

Usage:
    python -m evaluation.run_eval                    # Run all entries
    python -m evaluation.run_eval --category single_company
    python -m evaluation.run_eval --limit 5          # Quick smoke test
    python -m evaluation.run_eval --output report.json
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure backend is on the path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.evaluation.golden_dataset import (
    GoldenReport,
    load_golden_dataset,
    run_golden_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parent / "reports"


def print_report(report: GoldenReport) -> None:
    """Print a formatted evaluation report to stdout."""
    print("\n" + "=" * 70)
    print("  GOLDEN DATASET EVALUATION REPORT")
    print("=" * 70)
    print(f"  Date:              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total entries:     {report.total}")
    print(f"  Passed:            {report.passed}")
    print(f"  Failed:            {report.failed}")
    print(f"  Errors:            {report.errors}")
    print(f"  Pass rate:         {report.passed / max(report.total, 1):.1%}")
    print(f"  Avg overall score: {report.avg_overall_score:.3f}")
    print(f"  Avg confidence:    {report.avg_confidence:.3f}")
    print(f"  Avg latency:       {report.avg_latency:.1f}s")
    print("-" * 70)

    if report.by_category:
        print("\n  BY CATEGORY:")
        for cat, stats in sorted(report.by_category.items()):
            print(
                f"    {cat:20s}  "
                f"pass={stats['passed']}/{stats['total']}  "
                f"rate={stats['pass_rate']:.0%}  "
                f"avg={stats['avg_score']:.3f}"
            )

    print("\n  INDIVIDUAL RESULTS:")
    print(f"  {'ID':<14} {'Score':>6} {'Conf':>6} {'Time':>6} {'Status':<8} Query")
    print("  " + "-" * 66)
    for r in report.results:
        status = "ERROR" if r.error else ("PASS" if r.passed else "FAIL")
        print(
            f"  {r.entry_id:<14} "
            f"{r.overall_score:>5.3f} "
            f"{r.confidence:>5.3f} "
            f"{r.latency_seconds:>5.1f}s "
            f"{status:<8} "
            f"{r.query[:40]}..."
        )

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run golden dataset evaluation")
    parser.add_argument(
        "--category",
        type=str,
        choices=["single_company", "cross_company", "multi_step", "footnote", "risk_factor"],
        help="Filter by query category",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of entries to run",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Pass threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON report file path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to golden_dataset.json",
    )
    args = parser.parse_args()

    # Load dataset
    dataset_path = Path(args.dataset) if args.dataset else None
    entries = load_golden_dataset(dataset_path)

    if not entries:
        print("No golden dataset entries found. Exiting.")
        sys.exit(1)

    # Filter by category
    categories = [args.category] if args.category else None

    # Limit entries
    if args.limit and args.limit < len(entries):
        entries = entries[: args.limit]

    print(f"Running evaluation on {len(entries)} entries...")
    start = time.time()

    report = run_golden_dataset(
        entries=entries,
        pass_threshold=args.threshold,
        categories=categories,
    )

    elapsed = time.time() - start
    print_report(report)
    print(f"\nTotal wall time: {elapsed:.1f}s")

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(args.output)
        if args.output
        else REPORTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    report_data = report.to_dict()
    report_data["timestamp"] = datetime.now().isoformat()
    report_data["wall_time_seconds"] = round(elapsed, 2)

    output_path.write_text(json.dumps(report_data, indent=2))
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
