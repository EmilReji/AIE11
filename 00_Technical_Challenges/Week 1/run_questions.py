"""Run a question file through the existing CLI and save one report per run."""

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

PROJECT_DIR = Path(__file__).resolve().parent


def output_text(value: str | bytes | None) -> str:
    """TimeoutExpired may hold bytes even when text mode was requested."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "questions", type=Path, help="UTF-8 file: one question per line"
    )
    parser.add_argument("--corpus", type=Path, help="optional corpus file")
    parser.add_argument("-k", type=int, default=3, help="retrieved chunks (default: 3)")
    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="maximum seconds per question (default: 300)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "logs" / "evaluations",
        help="report directory (default: Week 1/logs/evaluations)",
    )
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("-k must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a finite positive number")

    question_path = args.questions.resolve()
    try:
        questions = [
            line.strip()
            for line in question_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError) as error:
        parser.error(f"Cannot read questions: {error}")
    if not questions:
        parser.error("The question file must contain at least one nonblank line")

    # Resolve caller-relative paths before launching children from the project directory.
    if args.corpus:
        corpus = args.corpus.resolve()
    else:
        candidates = sorted((PROJECT_DIR / "data").glob("*.txt"))
        if len(candidates) != 1:
            parser.error("Expected one .txt in data/; select a file with --corpus")
        corpus = candidates[0]
    if not corpus.is_file():
        parser.error(f"Corpus file does not exist: {corpus}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    # Exclusive creation plus a random suffix prevents collisions and overwrites.
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f"evaluation_{timestamp}_",
        suffix=".txt",
        dir=args.output_dir.resolve(),
        delete=False,
    ) as report:
        print(f"Report: {report.name}", flush=True)
        report.write(
            f"Run (UTC): {timestamp}\nQuestions file: {question_path}\n"
            f"Corpus: {corpus}\nQuestions: {len(questions)}\n"
            f"k: {args.k}\nTimeout per question: {args.timeout}s\n"
            "Chunk display: CLI previews (up to 400 characters each)\n"
            "Execution success does not imply answer correctness.\n"
        )
        failures = 0
        for number, question in enumerate(questions, start=1):
            print(f"Running question {number}/{len(questions)}...", flush=True)
            report.write(
                f"\n{'=' * 72}\nQuestion {number}/{len(questions)}\nInput: {question}\n"
            )
            report.flush()
            command = [
                sys.executable,
                "-u",
                "-m",
                "retrieval_app",
                "--corpus",
                str(corpus),
                "-k",
                str(args.k),
                "--show-chunks",
                "--",
                question,
            ]
            started = time.perf_counter()
            try:
                result = subprocess.run(
                    command,
                    cwd=PROJECT_DIR,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=args.timeout,
                )
                status = f"Exit status: {result.returncode}"
                stdout, stderr = result.stdout, result.stderr
                failures += result.returncode != 0
            except subprocess.TimeoutExpired as error:
                status = f"TIMEOUT after {args.timeout}s"
                stdout, stderr = output_text(error.stdout), output_text(error.stderr)
                failures += 1
            except OSError as error:
                status = "FAILED TO START"
                stdout, stderr = "", str(error)
                failures += 1
            except KeyboardInterrupt:
                report.write("INTERRUPTED by user; current output was not collected.\n")
                report.flush()
                print(f"\nStopped. Partial report: {report.name}")
                return 130

            report.write(
                f"{status}\nDuration: {time.perf_counter() - started:.2f}s\n"
                f"\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\n"
                "Manual result (PASS/FAIL):\nFailure stage and notes:\n"
            )
            report.flush()
            print(f"  {status}", flush=True)

        report.write(
            f"\nCompleted: {len(questions)}; execution failures/timeouts: {failures}\n"
        )
        print(
            f"Completed. Execution failures/timeouts: {failures}\nReport: {report.name}"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
