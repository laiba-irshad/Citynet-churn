# Candidate model comparison (Dummy baseline, Logistic Regression, Random Forest) on CityNet.
# Run: python src/model_candidates.py

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from common import load_split
from lab3_pipeline import make_preprocess


def get_candidates():
    return {
        "baseline": DummyClassifier(strategy="most_frequent"),
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
        ),
    }


def evaluate_candidates(X_train, y_train, candidates=None, cv_folds=5):
    if candidates is None:
        candidates = get_candidates()

    scoring = {
        "pr_auc": "average_precision",
        "recall": "recall",
        "precision": make_scorer(precision_score, zero_division=0),
    }

    results = {}
    for name, model in candidates.items():
        pipe = Pipeline([
            ("pre", make_preprocess()),
            ("model", model),
        ])
        cv = cross_validate(pipe, X_train, y_train, cv=cv_folds, scoring=scoring)
        metrics = {
            m: round(float(cv[f"test_{m}"].mean()), 3)
            for m in ("pr_auc", "recall", "precision")
        }
        results[name] = metrics
        print(
            f"{name:10s} | "
            f"pr_auc: {metrics['pr_auc']:.3f} | "
            f"recall: {metrics['recall']:.3f} | "
            f"precision: {metrics['precision']:.3f}"
        )

    return results


if __name__ == "__main__":
    X_train, _, _, y_train, _, _ = load_split()
    evaluate_candidates(X_train, y_train)