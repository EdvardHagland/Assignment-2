from __future__ import annotations

import argparse
import datetime as dt
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def log(message: str) -> None:
    timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def run_step(command: list[str], cwd: Path) -> None:
    log(f"Running: {' '.join(command)}")
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(cwd))
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        log(f"Step failed after {elapsed:.1f}s: {' '.join(command)}")
        raise SystemExit(completed.returncode)
    log(f"Finished in {elapsed:.1f}s: {' '.join(command)}")


def clean_outputs(project_root: Path) -> None:
    artifacts_dir = project_root / "artifacts"
    report_html = project_root / "report" / "assignment2_report.html"

    if artifacts_dir.exists():
        log(f"Removing stale artifacts directory: {artifacts_dir}")
        shutil.rmtree(artifacts_dir)
    if report_html.exists():
        log(f"Removing stale report file: {report_html}")
        report_html.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the consultation pipeline in a Colab-friendly way.")
    parser.add_argument("--config", default="config/project_config.yml")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-gemini", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    log("Loading environment variables from .env if present")
    load_env(project_root / ".env")

    if args.clean:
        clean_outputs(project_root)

    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 2))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config_arg = ["--config", args.config]
    log("Starting pipeline")
    run_step([sys.executable, "scripts/prepare_corpus.py", *config_arg], project_root)
    run_step([sys.executable, "scripts/embed_cluster.py", *config_arg], project_root)
    if args.skip_validation:
        log("Skipping validation stage")
    else:
        run_step([sys.executable, "scripts/validate_clusters.py", *config_arg], project_root)
    if not args.skip_gemini:
        log("Starting Gemini interpretation stage")
        run_step(
            [
                sys.executable,
                "scripts/gemini_interpret.py",
                *config_arg,
                "--batch-size",
                "1",
                "--examples-per-cluster",
                "5",
                "--temperature",
                "0.0",
                "--max-output-tokens",
                "7000",
                "--raw-dir",
                "artifacts/raw/gemini",
            ],
            project_root,
        )
    else:
        log("Skipping Gemini interpretation stage")
    log("Starting cluster inspection stage")
    run_step([sys.executable, "scripts/inspect_clusters.py", *config_arg], project_root)

    if not args.skip_render:
        log("Starting HTML render stage")
        run_step(
            [
                "Rscript",
                "-e",
                "rmarkdown::render('report/assignment2_report_refined.qmd', output_file='assignment2_report.html')",
            ],
            project_root,
        )
    else:
        log("Skipping HTML render stage")

    log("Pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
