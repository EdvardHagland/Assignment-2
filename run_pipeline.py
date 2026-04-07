from __future__ import annotations

import argparse
import os
import subprocess
import sys
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


def run_step(command: list[str], cwd: Path) -> None:
    print(f"\n=== Running: {' '.join(command)} ===", flush=True)
    completed = subprocess.run(command, cwd=str(cwd))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the consultation pipeline in a Colab-friendly way.")
    parser.add_argument("--config", default="config/project_config.yml")
    parser.add_argument("--skip-gemini", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    load_env(project_root / ".env")

    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 2))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    config_arg = ["--config", args.config]
    run_step([sys.executable, "scripts/prepare_corpus.py", *config_arg], project_root)
    run_step([sys.executable, "scripts/embed_cluster.py", *config_arg], project_root)
    run_step([sys.executable, "scripts/validate_clusters.py", *config_arg], project_root)
    if not args.skip_gemini:
        run_step(
            [
                sys.executable,
                "scripts/gemini_interpret.py",
                *config_arg,
                "--batch-size",
                "1",
                "--examples-per-cluster",
                "5",
                "--max-output-tokens",
                "3500",
            ],
            project_root,
        )
    run_step([sys.executable, "scripts/inspect_clusters.py", *config_arg], project_root)

    if not args.skip_render:
        run_step(
            [
                "Rscript",
                "-e",
                "rmarkdown::render('report/assignment2_report_refined.qmd', output_file='assignment2_report.html')",
            ],
            project_root,
        )

    print("\nPipeline complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
