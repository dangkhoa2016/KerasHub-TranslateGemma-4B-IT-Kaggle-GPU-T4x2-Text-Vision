# Kaggle usage

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](KAGGLE.vi.md)

## Importing the notebook

Use Kaggle's notebook import dialog and choose the **GitHub** source. Select this repository and import `notebooks/kaggle-t4x2-text-vision.ipynb`.

The notebook itself contains all orchestration cells. Its first cell clones or refreshes the repository under `/kaggle/working`, which gives the remaining cells a normal source tree to execute.

A shell command such as `git clone` only creates files in the notebook runtime; it does not make Kaggle replace the open notebook with another `.ipynb`. That is why GitHub notebook import and repository cloning are used together.

## Required session settings

- Accelerator: **GPU T4 x2**
- Internet: **On**
- Model input: Keras TranslateGemma with the `translategemma_4b_it` preset

## Re-running after repository updates

Re-run the first notebook cell. When a Git working copy already exists, the cell fetches `origin/main` and resets the working tree to that revision. Generated local runtime files remain ignored by Git; the notebook recreates `.env` from `.env.example` when necessary.

## Model discovery

The default mounted path is `/kaggle/input/models/keras/translategemma/keras/translategemma_4b_it/1`. If this exact version directory is unavailable, `scripts/setup.sh` searches attached version directories when `MODEL_AUTO_DISCOVER=true`.
