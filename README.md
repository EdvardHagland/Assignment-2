# Colab Consultation Pipeline

This repository runs directly in Google Colab. The repository is public, so it can be cloned into any Colab runtime without creating a separate fork.

## Run In Colab

Use the shared Colab notebook, or run the cells below in a fresh Colab runtime.

Before running the notebook, add these secrets in the Colab Secrets pane:

- `GEMINI_API_KEY`: required if you want the Gemini interpretation stage
- `HF_TOKEN`: optional, but recommended for faster Hugging Face downloads
- `GEMINI_MODEL`: optional override, defaults to `gemini-3.1-pro-preview`

### Cell 1: Secrets and runtime variables

```python
from google.colab import userdata
import os
import shutil
from pathlib import Path

REPO_DIR = Path("/content/Assignment-2")
CORPUS_ID = "12527"  # default example; other options are listed below
SKIP_VALIDATION = True  # set to False if you want the full validation stage

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

os.chdir("/content")
if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)
```

### Cell 2: Clone the repo and install Python dependencies

```bash
!git clone https://github.com/EdvardHagland/Assignment-2.git /content/Assignment-2
%cd /content/Assignment-2
!python -m pip install -r requirements.txt
```

### Cell 3: Install R and the report packages

```bash
!apt-get update
!apt-get install -y r-base
!Rscript -e "pkgs <- c('rmarkdown','knitr','dplyr','glue','jsonlite','readr','tibble','yaml','ggplot2'); need <- setdiff(pkgs, rownames(installed.packages())); if (length(need)) install.packages(need, repos='https://cloud.r-project.org')"
```

### Cell 4: Select the corpus and run the pipeline

```python
import os
import subprocess

os.chdir("/content/Assignment-2")

subprocess.run(
    ["python", "-u", "scripts/select_corpus.py", "--corpus", CORPUS_ID, "--clean"],
    check=True,
)

cmd = ["python", "-u", "run_pipeline.py", "--clean"]
if SKIP_VALIDATION:
    cmd.append("--skip-validation")
if not os.getenv("GEMINI_API_KEY"):
    cmd.append("--skip-gemini")

subprocess.run(cmd, check=True)

print("Done. Report written to /content/Assignment-2/report/assignment2_report.html")
```

### Bundled corpora

The notebook defaults to `12527`.

If you want a different bundled corpus, change `CORPUS_ID` in Cell 1 to one of these:

- `12527`: Requirements for Artificial Intelligence
- `16213`: European Open Digital Ecosystems
- `14031`: Fitness check of EU legislation on trade in seal products
- `1424`: European Defence Fund and EU Defence Industrial Development Programme

If `GEMINI_API_KEY` is missing, the pipeline still runs and simply skips the Gemini interpretation stage.

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
