# Colab Consultation Pipeline

This repo is meant to be run directly from Google Colab. Because the repo is public, nobody needs to fork it or create a new GitHub repo first. They can open a fresh Colab notebook, add their own secrets, paste one cell, and run the pipeline in their own session.

## Copy-Paste Into Colab

Before running the cell below, add these secrets in the Colab Secrets pane:

- `GEMINI_API_KEY`: required if you want the Gemini interpretation stage
- `HF_TOKEN`: optional, but recommended for faster Hugging Face downloads
- `GEMINI_MODEL`: optional override, defaults to `gemini-3.1-pro-preview`

Then paste this single cell into Google Colab:

```python
from google.colab import userdata
import os
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/EdvardHagland/Assignment-2.git"
REPO_DIR = Path("/content/Assignment-2")
CORPUS_ID = "16213"  # choose from: 12527, 16213, 14031, 1424
SKIP_VALIDATION = True

def get_secret(name, default=""):
    try:
        value = userdata.get(name)
        return value if value else default
    except Exception:
        return default

hf_token = get_secret("HF_TOKEN")
gemini_key = get_secret("GEMINI_API_KEY")
gemini_model = get_secret("GEMINI_MODEL", "gemini-3.1-pro-preview")

if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
if gemini_key:
    os.environ["GEMINI_API_KEY"] = gemini_key
if gemini_model:
    os.environ["GEMINI_MODEL"] = gemini_model

if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)

subprocess.run(["git", "clone", REPO_URL, str(REPO_DIR)], check=True)
os.chdir(REPO_DIR)

subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)

if shutil.which("Rscript") is None:
    subprocess.run(["apt-get", "update"], check=True)
    subprocess.run(["apt-get", "install", "-y", "r-base"], check=True)

subprocess.run(
    [
        "Rscript",
        "-e",
        "install.packages(c('rmarkdown','knitr','dplyr','glue','jsonlite','readr','tibble','yaml','ggplot2'), repos='https://cloud.r-project.org')",
    ],
    check=True,
)

subprocess.run(
    ["python", "-u", "scripts/select_corpus.py", "--corpus", CORPUS_ID, "--clean"],
    check=True,
)

cmd = ["python", "-u", "run_pipeline.py", "--clean"]
if SKIP_VALIDATION:
    cmd.append("--skip-validation")
if not gemini_key:
    cmd.append("--skip-gemini")

subprocess.run(cmd, check=True)

print(f"Done. Report written to: {REPO_DIR / 'report' / 'assignment2_report.html'}")
```

## Bundled Corpora

- `12527`: Requirements for Artificial Intelligence
- `16213`: European Open Digital Ecosystems
- `14031`: Fitness check of EU legislation on trade in seal products
- `1424`: European Defence Fund and EU Defence Industrial Development Programme

Change `CORPUS_ID` in the Colab cell to switch corpus. If `GEMINI_API_KEY` is missing, the cell still runs the quantitative pipeline and skips Gemini automatically.

## What The Pipeline Does

- prepares attachment-aware paragraph-like units
- embeds them with `BAAI/bge-m3`
- clusters the embedding space
- validates the discovered clustering
- extracts lexical and metadata signals
- optionally asks Gemini for comparative qualitative interpretation
- renders the HTML report

## Main Files

- `config/project_config.yml`: pipeline configuration
- `run_pipeline.py`: Colab-friendly launcher
- `scripts/select_corpus.py`: switch the active bundled corpus
- `scripts/prepare_corpus.py`: unit construction
- `scripts/embed_cluster.py`: embeddings, clustering, and figures
- `scripts/validate_clusters.py`: permutation/bootstrap validation
- `scripts/inspect_clusters.py`: term tables, metadata lift, and inspection outputs
- `scripts/gemini_interpret.py`: structured Gemini interpretation
- `report/assignment2_report_refined.qmd`: report source

## Notes

- The pipeline reads `data/compiled/consultation_compiled.csv`, which is updated by `scripts/select_corpus.py`.
- `artifacts/` are ignored on purpose, and the heavy Colab step is still embedding plus clustering.
- The default pipeline path now runs one corpus at a time and uses a single full-corpus Gemini interpretation pass.
