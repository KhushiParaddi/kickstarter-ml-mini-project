# Predicting Kickstarter Project Success

**UE24CS352A – Machine Learning Mini-Project | PES University**
**Team:** Lakshya Budhauliya, Khushi Paraddi

Binary classification of Kickstarter campaigns (successful vs failed) using **only information known at launch**. This is a re-implementation of Khosla, Reinecke and Wittenbrink, *"Crowdfunding: Predicting Kickstarter Project Success"*, with eight models compared.

## Results (test set, 20,771 projects)

| Model | Accuracy | F1 |
|---|---|---|
| **LightGBM (final model)** | **0.864** | 0.906 |
| Neural Net (MLP) | 0.864 | 0.911 |
| Random Forest | 0.852 | 0.898 |
| Logistic Regression | 0.831 | 0.881 |
| LinearSVC | 0.828 | 0.878 |
| Lasso | 0.827 | 0.877 |
| Ridge | 0.824 | 0.875 |
| OLS | 0.823 | 0.874 |

Ablation (LightGBM, 3 seeds): metadata only 0.857 accuracy, all features 0.864, so text features add about 0.7 points.

## Repository contents

```
.
├── Kickstarter_Success_Prediction.ipynb   # full pipeline: load, clean, EDA, features, 8 models, ablation, demo
├── kickstarter_model.joblib               # saved feature builder + LightGBM model (created by the notebook)
├── Writeup.pdf                            # two-page project write-up
├── Presentation.pptx                      # review slide deck
└── README.md
```

## Dataset

The dataset is **not stored in this repo** (size). Use one of:

- **Web Robots "Kickstarter Datasets"** CSV zip (used for our results; includes the blurb text), or
- **Kaggle "Kickstarter Projects"** zip (no blurb, so the project title is used as the text field).

The notebook detects which schema you upload.

## How to run

### Option A: Google Colab (recommended)

1. Open `Kickstarter_Success_Prediction.ipynb` in Google Colab.
2. Click **Runtime > Run all**.
3. When prompted, **upload the dataset zip**. Alternatively, set `DATA_URL` or `DATA_PATH` in the config cell to skip the upload.
4. The last cell launches a Gradio demo and prints a public link.

### Option B: Local

```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm vaderSentiment gradio joblib
jupyter notebook Kickstarter_Success_Prediction.ipynb
```

The notebook's file-upload cell is Colab-specific. Locally, set `DATA_PATH` in the config cell to your zip or CSV path.

## Configuration (first code cell)

| Variable | Default | Meaning |
|---|---|---|
| `MAX_ROWS` | 100,000 | Cap on rows used; `None` uses everything (slower) |
| `MIN_COUNT` | 30 | Minimum projects per category level for target encoding (paper uses 1000 on 155k rows) |
| `N_TOPICS` | 20 | Number of LDA topics |
| `RUN_TUNING` | `False` | Set `True` to run a randomized hyperparameter search for LightGBM |
| `SEED` | 42 | Random seed |

Full run takes roughly 5 to 10 minutes on Colab (feature building about 2 minutes, model training about 1.5 minutes).

## Method summary

1. **Clean:** keep successful/failed projects, deduplicate, convert goal to USD, drop post-launch fields (backers, pledged, staff pick).
2. **Features (241):** 219 metadata (goal, duration, prep time, launch month/weekday, text length, out-of-fold target encodings, one-hot categories), 1 out-of-fold Naive Bayes text probability, 1 VADER sentiment score, 20 LDA topic probabilities.
3. **Models:** OLS, Ridge, Lasso (thresholded at 0.5), Logistic, LinearSVC, MLP, Random Forest, LightGBM, with class weighting where supported.
4. **Evaluate:** 70/30 stratified split, accuracy, F1, per-class precision and recall, confusion matrix, feature importance, ablation.
5. **Demo:** Gradio app that predicts the probability of success from launch-time inputs.

## Limitations

- Single train/test split, no cross-validation.
- Hyperparameter tuning is optional and was not run for the reported results.
- VADER replaces Stanford CoreNLP, and a data subset is used, so numbers are not directly comparable to the paper.
- The Gradio share link is temporary (about one week); re-run the last cell to get a new one.

## Reference

Khosla, Reinecke, Wittenbrink. *Crowdfunding: Predicting Kickstarter Project Success.* Stanford University.
