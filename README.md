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

### 4. Use the bundled compiled consultation CSV

This repo can include one ready-to-run compiled corpus at:

```text
data/compiled/consultation_compiled.csv
```

If you want to swap in another consultation, replace that file or edit `config/project_config.yml` to point somewhere else.

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
- `scripts/prepare_corpus.py`: unit construction
- `scripts/embed_cluster.py`: embeddings, clustering, and figures
- `scripts/validate_clusters.py`: permutation/bootstrap validation
- `scripts/inspect_clusters.py`: term tables, metadata lift, and inspection outputs
- `scripts/gemini_interpret.py`: structured Gemini interpretation
- `report/assignment2_report_refined.qmd`: report source

## Notes

- This repo is set up to optionally version one compiled consultation CSV at `data/compiled/consultation_compiled.csv`.
- `artifacts/` are ignored on purpose, and other `data/` contents stay ignored.
- On Colab, the heavy step is embedding. That is the main reason to run this version in the cloud.
