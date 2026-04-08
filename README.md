# Colab Consultation Pipeline

This repo is a clean clone target for Google Colab. It keeps the current consultation-text pipeline, but strips it down to the files you actually need to run heavy embedding work in the cloud.

## Quick start

### 1. Create a new GitHub repo and push this folder

From the folder root:

```bash
git init
git add .
git commit -m "Initial Colab pipeline"
git branch -M main
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```

### 2. In Google Colab, clone the repo

```python
!git clone https://github.com/<your-user>/<your-repo>.git
%cd <your-repo>
```

### 3. Install dependencies

```python
!pip install -r requirements.txt
```

### 4. Choose a bundled compiled corpus

This repo now bundles multiple ready-to-run compiled corpora in:

```text
data/compiled/
```

The active pipeline target remains:

```text
data/compiled/consultation_compiled.csv
```

Use the selector script to point that active file at one of the bundled corpora:

```python
!python scripts/select_corpus.py --list
!python scripts/select_corpus.py --corpus 16213 --clean
```

Bundled corpora:
- `16213`: European Open Digital Ecosystems
- `14031`: Fitness check of EU legislation on trade in seal products
- `1424`: European Defence Fund and EU Defence Industrial Development Programme

### 5. Add your tokens

Create a `.env` file from `.env.example`, or set env vars in Colab:

```python
import os
os.environ["HF_TOKEN"] = "your_hf_token"
os.environ["HUGGINGFACE_HUB_TOKEN"] = "your_hf_token"
os.environ["GEMINI_API_KEY"] = "your_gemini_key"
os.environ["GEMINI_MODEL"] = "gemini-3.1-pro-preview"
```

### 6. Run the pipeline

Quantitative pipeline only:

```python
!python run_pipeline.py --skip-gemini --skip-render
```

Skip validation for a faster exploratory run:

```python
!python run_pipeline.py --skip-validation --skip-render
```

With Gemini:

```python
!python run_pipeline.py --skip-render
```

If you also install R and `rmarkdown`, you can run the full render:

```python
!python run_pipeline.py
```

### 7. Try another bundled corpus

Switch the active corpus, rerun the pipeline, and render again:

```python
!python scripts/select_corpus.py --corpus 14031 --clean
!python run_pipeline.py --clean --skip-validation
```

If you want to test several bundled corpora in one Colab session and keep each HTML output:

```python
import os
import shutil
import subprocess

os.makedirs("report/output", exist_ok=True)

for corpus_id in ["16213", "14031", "1424"]:
    subprocess.run(
        ["python", "scripts/select_corpus.py", "--corpus", corpus_id, "--clean"],
        check=True,
    )
    subprocess.run(
        ["python", "run_pipeline.py", "--clean", "--skip-validation"],
        check=True,
    )
    shutil.copyfile(
        "report/assignment2_report.html",
        f"report/output/{corpus_id}_assignment2_report.html",
    )
```

## What this repo does

The pipeline:
- prepares attachment-aware paragraph-like units
- embeds them with `BAAI/bge-m3`
- clusters the embedding space
- validates the discovered clustering
- extracts lexical and metadata signals
- optionally asks Gemini for comparative qualitative interpretation
- optionally renders the HTML report

## Main files

- `config/project_config.yml`: pipeline configuration
- `run_pipeline.py`: simple Colab-friendly launcher
- `scripts/select_corpus.py`: switch the active bundled corpus in Colab
- `scripts/prepare_corpus.py`: unit construction
- `scripts/embed_cluster.py`: embeddings, clustering, and figures
- `scripts/validate_clusters.py`: permutation/bootstrap validation
- `scripts/inspect_clusters.py`: term tables, metadata lift, and inspection outputs
- `scripts/gemini_interpret.py`: structured Gemini interpretation
- `report/assignment2_report_refined.qmd`: report source

## Notes

- This repo now bundles multiple compiled consultation CSVs and uses `data/compiled/consultation_compiled.csv` as the active alias that the pipeline reads.
- `artifacts/` are ignored on purpose, and other `data/` contents stay ignored.
- On Colab, the heavy step is embedding. That is the main reason to run this version in the cloud.
