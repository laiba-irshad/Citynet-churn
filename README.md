# CityNet churn baseline

## User story
As the retention manager, I want a weekly list of the 100 customers most likely to cancel next month, so my team can call them first[cite: 4].

## How to run
    pip install -r requirements.txt
    python src/lab7_save.py                       # train and save models/churn_pipeline.joblib[cite: 4]
    python src/predict.py data/new_customers.csv  # writes reports/predictions.csv[cite: 4]

## Results (validation, then test once)
| Metric | Baseline | Our model |
|---|---|---|
| PR-AUC | Random / baseline | 0.396 (validation) / 0.361 (test)[cite: 1, 3] |
| Churners in top 100 | ~18 (random 100 calls)[cite: 1] | 49 (validation, ~2.7x lift)[cite: 1] |

- **Model & Pipeline:** Preprocessing pipeline (`make_preprocess()`) paired with `LogisticRegression(max_iter=1000, class_weight="balanced")`, serialized to `models/churn_pipeline.joblib` via `joblib.dump` and loaded in `src/predict.py` via `joblib.load` to score incoming customers using `predict_proba`[cite: 4].
- **Validation metrics at threshold `0.5`:** Precision `0.305`, Recall `0.687`, Confusion Matrix `[[541, 280], [56, 123]]` (lowering threshold to `0.3` catches more churners at the cost of many more false positives)[cite: 1].
- **Top permutation features:** `tenure` (`0.13`), `auto_pay` (`0.044`), `support calls` (`0.043`), and `paid late` (`0.026`)[cite: 3].

## Intended use
Weekly ranking of customer records (`data/new_customers.csv`) by predicted churn risk (`churn_risk`, sorted descending) in `reports/predictions.csv` so the retention call team can prioritize outreach to the top 100 highest-risk customers[cite: 1, 4].

## Not for
- Automated account termination, billing penalties, or unreviewed punitive actions against flagged customers.
- Claiming exact causal proof of why an individual customer intends to leave without direct outreach.
- Applying directly to different telecom markets, pricing models, or regions outside the CityNet training distribution without retraining.

## Assumptions
- Input files passed to `src/predict.py` match the feature schema and data types expected by the saved preprocessing and logistic regression pipeline[cite: 4].
- Historical relationships between features (such as tenure, auto-pay, support calls, and late payments) and churn remain consistent enough for weekly prioritization[cite: 3, 4].
- The retention team works through a ranked list (such as top 100) rather than relying solely on a rigid probability cutoff[cite: 1, 4].

## Risks and limitations
- **Misses quiet long-tenure churners:** Caught churners average `18.9` months tenure (`8.6 h` outage), whereas missed churners average `43.2` months tenure (`6.1 h` outage)—long-time subscribers who leave quietly with lower outage signals are harder to catch[cite: 3].
- **Uneven regional catch rates:** Churner catch rate varies across regions—`Central` (`0.76`), `North` (`0.73`), `West` (`0.71`), and `South` (`0.68`) perform stronger than `East` (`0.50`), where only half of churners are caught[cite: 3].
- **False positives vs. false negatives trade-off:** At threshold `0.5`, precision is `0.305` (many non-churners flagged), and lowering the threshold to `0.3` increases false calls even further[cite: 1].
- **Validation-to-test gap:** Test PR-AUC (`0.361`) is slightly lower than validation PR-AUC (`0.396`), so ongoing monitoring is needed as new customer batches are scored[cite: 1, 3, 4].