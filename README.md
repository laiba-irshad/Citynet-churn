# CityNet churn baseline

## User story
As the retention manager, I want a weekly list of the 100 customers most likely to cancel next month, so my team can call them first.

## How to run
    pip install -r requirements.txt
    python src/lab7_save.py                       # train and save the pipeline
    python src/predict.py data/new_customers.csv  # writes reports/predictions.csv

## Results (validation, then test once)
| Metric | Baseline | Our model |
|---|---|---|
| PR-AUC | 0.179 (dummy / random) | 0.396 (validation) / 0.361 (test) |
| Churners in top 100 | 18 (random 100 calls) | 49 (validation, ~2.7x lift) |

## Intended use
Weekly ranking of active CityNet subscribers by predicted churn risk (`predict_proba`) so the retention call team can prioritize outreach to the top 100 highest-risk customers.

## Not for
- Automatic account termination, penalty billing, or punitive actions without human review.
- Causal attribution (determining *why* an individual customer is unhappy without speaking to them).
- Use on telecom markets, regions, or billing structures outside the historical CityNet dataset.

## Assumptions
- Incoming customer records (`data/new_customers.csv`) follow the exact same schema, feature definitions, and preprocessing distribution as the training data (`tenure_months`, `monthly_bill`, `support_calls_90d`, `outage_hours_90d`, `auto_pay`, `paid_late_last_bill`, `region`, `plan`, `tv_bundle`, `payment_method`, `age`).
- Customer behavior and churn drivers remain stable over the weekly prediction horizon.
- Retention outreach capacity is ~100 calls per week, making top-K ranking more actionable than a fixed probability cutoff alone.

## Risks and limitations
- **Misses quiet long-tenure churners:** Error analysis shows caught churners average `18.9` months tenure (`8.6` outage hours), whereas missed churners average `43.2` months tenure (`6.1` outage hours)—long-time customers who cancel without recent outages or support calls are under-detected.
- **Regional performance gap:** Detection is uneven across regions; churners in the **East** region are caught only `50%` of the time (`0.50`), compared to `0.76` in **Central** and `0.73` in **North**.
- **False positives on threshold classification:** At threshold `0.5`, validation precision is `0.305` (recall `0.687`), meaning many flagged customers would not have churned; lowering the threshold to `0.3` increases recall further but causes many more unnecessary calls.
- **Generalization drop:** Performance drops slightly from validation PR-AUC (`0.396`) to unseen test PR-AUC (`0.361`), requiring ongoing monitoring for data drift.