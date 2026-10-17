# Climate-Sensitive Death Prediction: Technical Report

October 2026

## Summary

The final model scores 0.8249 on the competition metric (0.6 × F1 + 0.4 × ROC-AUC) in village-grouped out-of-fold validation. F1 is 0.8214 and ROC-AUC is 0.8303 at a decision threshold of 0.31.

The model averages two class-balanced Random Forests. Each one re-ranks adults aged 18 to 59 with a small logistic regression on gender and death year. One of the two forests sees no climate features.

The main finding: climate data adds no measurable signal. Age and death year carry the prediction. A logistic regression on those two variables alone scores 0.8121. Tuning moves the score to 0.8195. Eleven further ideas failed to beat the tuned baseline under leak-free testing.

## Problem and data

The task: predict whether a recorded death in eastern Uganda is climate-sensitive (is_climate_sensitive = 1). The challenge scores 0.6 × F1 on hard labels plus 0.4 × ROC-AUC on probabilities. A submission carries both columns, TargetF1 and TargetRAUC.

| File | Rows | Columns | Contents |
| --- | --- | --- | --- |
| Train.csv | 3,146 | 13 | Deaths with demographics, weather, coordinates, location and the target |
| Test.csv | 1,030 | 12 | Same fields, no target |
| climate_features.csv | 4,176 | 18 | Satellite and reanalysis features per ID: rainfall and temperature windows (7, 30, 90 days), NDVI, elevation, slope |
| SampleSubmission.csv | 1,030 | 3 | ID, TargetF1, TargetRAUC |

After the merge and removal of 2 duplicate rows, the training frame holds 3,144 rows and 29 columns, with no missing values. The positive rate is about 65%.

## What the data shows

Age dominates. Age has a correlation of -0.44 with the target, the strongest of any column. The effect is U-shaped, not linear: the rate peaks at 92% for ages 1 to 4 and falls to 30% for ages 60 and over (table below).

| Age band | Deaths | Climate-sensitive rate |
| --- | --- | --- |
| 0 (infant) | 1,110 | 79% |
| 1 to 4 | 621 | 92% |
| 5 to 17 | 139 | 63% |
| 18 to 39 | 339 | 52% |
| 40 to 59 | 289 | 48% |
| 60+ | 646 | 30% |
| All ages | 3,144 | 65% |

Infant deaths sit 14 points below ages 1 to 4, a sign of a different cause mix. A single linear age term misses this shape, so the model uses age bands and tree splits.

- Death year is the second signal (r = -0.17). The positive rate falls from about 0.78 in 2008 to 2011 to about 0.50 in 2014 to 2021. The decline holds inside every age band (age 0: r = -0.73, p = 0.002, and 18 to 39: r = -0.86, p = 0.0004), so age mix does not explain the trend. Malaria-control scale-up in the region over the same years fits the pattern.
- From 2019 the data shifts toward under-5 deaths. Median age drops to 0 and the under-5 share rises from between 40% and 55% to between 70% and 84%, in train and test alike. This points to a change in surveillance focus.
- Gender has a small effect (chi-square p = 0.021), concentrated in ages 5 to 59: 62% positive for women against 46% for men.
- Zone and region show no effect (p = 0.375 and p = 0.964).
- Every climate column, raw or engineered, has an absolute correlation below 0.07 with the target. hot_days_30d is 0 for every row.
- The temperature family and the rainfall family each correlate internally at r = 0.6 to 0.9. The 7, 30 and 90 day windows repeat the same information.
- Geography does not transfer. 10 of 11 test locations and 3 test regions (Kalangala, Kamuli, Sembabule) never appear in training. Elevation, slope, latitude and longitude show the largest train-test shift (Kolmogorov-Smirnov statistics of 0.30 to 0.40).

## Evaluation

All scores in this report come from cross-validation grouped by village, so no village appears in both the training and the scoring fold. This matches the test set, where most villages are new.

The first notebook version had four defects, each fixed before the final model:

1. RandomizedSearchCV returned NaN for every candidate. The needs_proba argument no longer exists in current scikit-learn. The fix uses response_method and error_score="raise", so failures surface.
2. Permutation-importance pruning chose features on the same fold used to score them. The apparent gain vanished once fixed: pruning scored 0.0011 worse (p = 0.017).
3. PCA and scaling ran on all rows before any split. A custom FamilyPCACompressor transformer now fits inside each training fold only.
4. A shared, uncloned ColumnTransformer leaked between age-segmented models.

The tuned model went through nested cross-validation: the hyperparameter search ran again inside each of 15 outer folds. Model comparisons use the corrected resampled t-test, which accounts for overlap between folds. The combined metric picks F1 at the best threshold on a 0.20 to 0.70 grid, so the search optimises the score the leaderboard reports.

## Final model

The model has three parts.

1. Features. A single build_frame() function cleans train and test the same way. Location names get typo fixes. Dates give year, month, rainy season (March to May, September to November) and a three-level era (up to 2010, 2011 to 2016, 2017 on). Age gives a six-level band plus flags for working-age women, rural elderly and peri-urban middle-aged deaths. Rainfall and temperature families compress to 3 PCA components each. Region, coordinates, elevation, slope and hot_days_30d are dropped. Location serves only to group folds. The model never sees location.
2. Random Forest with adult re-ranking. A class-balanced Random Forest scores every death. Inside ages 18 to 59 the forest ranks poorly (AUC about 0.60), while gender and death year reach about 0.63. A logistic regression on those two variables reorders adults. Each adult then takes a score from the forest's own adult score distribution, so adults keep their position against other age groups.
3. Averaging. The final prediction averages two such models: one with all features and one without climate features.

The tuned hyperparameters: 300 trees, maximum depth 6, at least 8 samples per leaf, at least 5 samples per split, 50% of features per split. Search scores ranged from 0.8109 to 0.8242. Nested cross-validation put the tuned forest at 0.8195 ± 0.0040, ahead of the age plus year baseline (0.8121, p = 0.042). Tuned LightGBM (0.8197) and HistGradientBoosting (0.8188) tied with the forest, so the simpler model stayed.

The final fit uses all 3,144 training rows. The submission predicts 77.1% of test deaths as climate-sensitive.

## Robustness check

The averaged model holds up best on districts the training data lacks. The public leaderboard scores about 30% of the test set, so the model choice targets the hidden 70%. 371 of 1,030 test rows (36%) come from four districts absent from training: Kalangala, Kamuli, Sembabule and Mitooma. In those rows, up to 39% of values for some climate features fall outside the training range.

Each candidate used the same hyperparameters. The held-out district score leaves out each of the 3 districts outside the original Iganga and Mayuge study site in turn and scores the one left out, with the threshold fixed from village cross-validation.

| Model | Village CV | Held-out districts | Era up to 2010 | Era 2011 to 2016 | Era 2017 on |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 0.8209 | 0.8126 | 0.8520 | 0.8508 | 0.6928 |
| RF + adult re-ranking | 0.8240 | 0.8122 | 0.8544 | 0.8494 | 0.7020 |
| Adult re-ranking, RF without climate | 0.8250 | 0.8222 | 0.8530 | 0.8506 | 0.7035 |
| Average of the two above (final) | 0.8255 | 0.8230 | 0.8559 | 0.8514 | 0.7017 |

The full-feature forest drops most on held-out districts (0.8126). The forest splits on climate values and new districts push those values out of range. The averaged model halves the exposure. None of these gaps reach significance alone, and the held-out check covers 339 rows from 3 districts. Treat the choice as a hedge, not a proven gain.

## Experiments left out

Each idea below ran against the tuned Random Forest (0.8195) under village-grouped cross-validation (5 folds × 3 seeds) with the corrected resampled t-test. None improved the model.

| Idea | Score vs baseline | Reason for failure |
| --- | --- | --- |
| Humidity and feels-like temperature (Open-Meteo) | +0.0002, p = 0.60 | Repeats the existing temperature data |
| Humidity and feels-like temperature lagged 1 to 14, 15 to 45, 46 to 90 days | 0.8181, p = 0.34 | Raw correlation peaks at 15 to 45 days (r = 0.078), in line with malaria's lag, but tracks the year trend death_year already captures |
| WorldClim bioclimatic normals | +0.0004, p = 0.48 | All villages sit in a small region, so long-term normals barely vary |
| Location-month climate anomalies | 0.8180, p = 0.42 | Unseen test villages fall back to zone and month averages |
| Age × time interactions (4 variants) | best 0.8199, p = 0.78 | The forest finds these splits on its own |
| Death clustering per village (7, 14, 30 days) | 0.8173, p = 0.41 | Correlates with death_year (r = -0.31) |
| Post-2019 flag | 0.8183, p = 0.46 | The forest splits death_year at 2019 already |
| Permutation-importance pruning | -0.0011, p = 0.017 | Worse. The earlier gain came from fold leakage |
| Separate models for under-5 and 5+ | 0.8032, p = 0.004 | Worse. Each segment has too little data |
| Stacking and voting across model families | below the single model | Weaker models add noise |
| LightGBM, HistGradientBoosting, CatBoost | 0.8197, 0.8188, screen only | Tied with the Random Forest |

## Limits and next steps

The remaining error sits in ages 5 to 59 after 2017. This subgroup scores 0.606 on the combined metric with an AUC of 0.568, close to chance, against 0.832 for all other rows. Age × era interactions (0.610) and location-month anomalies (0.603) left the subgroup unchanged. The current covariates do not separate causes of death for this group.

- Cause-of-death covariates such as verbal autopsy symptoms or facility records would address the ceiling directly.
- Rows from more districts would firm up the held-out district check, which now rests on 339 rows.
- Confirm the 2019 surveillance shift with the data provider before relying on death_year in future years.
