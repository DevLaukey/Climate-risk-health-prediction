# Climate-Sensitive Death Prediction

Predict whether a recorded death in eastern Uganda is climate-sensitive, from demographics, location, weather and satellite climate features.

The challenge scores submissions on 0.6 × F1 + 0.4 × ROC-AUC. The final model scores 0.8249 in village-grouped out-of-fold validation (F1 0.8214, ROC-AUC 0.8303, threshold 0.31).

Read [REPORT.md](REPORT.md) for the full technical write-up.

## Key findings

- Age drives the prediction. The climate-sensitive rate peaks at 92% for ages 1 to 4 and falls to 30% for ages 60 and over.
- Death year is the second signal. The positive rate falls from about 0.78 (2008 to 2011) to about 0.50 (2014 to 2021).
- Climate features add no measurable signal over age and death year. Eleven climate and modelling ideas failed to beat the tuned baseline.
- 36% of test rows come from districts absent from training, so the final model limits its reliance on climate and location features.

## Project structure

| File | Purpose |
| --- | --- |
| climate-risk-health-prediction-simple.ipynb | Final notebook: data loading, EDA, model, robustness checks, submission |
| climate-risk-health-prediction.ipynb | Earlier working notebook with the first models and feature experiments |
| climate_health_starter_notebook_.ipynb | Starter notebook from the challenge |
| Train.csv | 3,146 training deaths with the target column is_climate_sensitive |
| Test.csv | 1,030 test deaths without the target |
| climate_features.csv | Extra climate features per ID: rainfall and temperature windows, NDVI, elevation, slope |
| data_dictionary.csv | Column descriptions for Train.csv and Test.csv |
| downloaded_climate_features_data_dictionary.csv | Column descriptions for climate_features.csv |
| SampleSubmission.csv | Submission format: ID, TargetF1, TargetRAUC |
| starter_submission.csv | Submission from the starter notebook |
| submission.csv | Submission from the final model |
| REPORT.md | Technical report |
| requirements.txt | Pinned Python dependencies |

## Setup

The project runs on Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce the submission

```bash
jupyter notebook climate-risk-health-prediction-simple.ipynb
```

Run all cells from top to bottom. The notebook writes submission.csv to the project folder. The hyperparameter search and the robustness checks take the longest steps. All random seeds are fixed at 42.

## Method

1. Merge Train.csv and Test.csv with climate_features.csv on ID and drop duplicate rows.
2. Clean location names and build features: age bands, death era, rainy season, rainfall and temperature summaries.
3. Drop features which do not transfer to unseen test locations: region, coordinates, elevation and slope.
4. Compress the temperature and rainfall families to 3 PCA components each, fit inside each training fold.
5. Score with a class-balanced Random Forest. Re-rank adults aged 18 to 59 with a logistic regression on gender and death year.
6. Average a model with climate features and a model without them.
7. Validate with cross-validation grouped by village, plus a leave-one-district-out check.

## Results

| Model | Village CV score |
| --- | --- |
| Logistic regression on age and death year (baseline) | 0.8121 |
| Tuned Random Forest (nested CV) | 0.8195 ± 0.0040 |
| Final averaged model | 0.8255 |

The remaining error sits in ages 5 to 59 after 2017, where the model ranks close to chance (AUC 0.568). REPORT.md covers this limit and the experiments left out.
