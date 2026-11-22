# Cheat / data-leakage check: temporarily add cancel_request_logged to NUM for Logistic Regression,
# observe PR-AUC (~0.909 indicates leakage), then remove it so NUM stays clean.
# Run: python src/leakage_check.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

import common
from common import load_split
from lab3_pipeline import make_preprocess

LEAK_COL = "cancel_request_logged"


def run_leakage_check(leak_col=LEAK_COL, cv_folds=5):
    X_train, _, _, y_train, _, _ = load_split()
    original_num = list(common.NUM)

    if leak_col not in X_train.columns:
        print(f"Column '{leak_col}' not found in X_train. Available columns: {list(X_train.columns)}")
        return

    # 1. Temporarily add the suspicious column to NUM and run Logistic Regression
    common.NUM = original_num + [leak_col]
    pipe = Pipeline([
        ("pre", make_preprocess()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    cv = cross_validate(pipe, X_train, y_train, cv=cv_folds, scoring="average_precision")
    leak_pr_auc = float(cv["test_score"].mean())

    # 2. Immediately restore NUM so the leakage column does not stay in features
    common.NUM = original_num

    print(f"With '{leak_col}' in NUM -> Logistic PR-AUC: {leak_pr_auc:.3f}")
    print("Warning: unusually high score indicates data leakage; column removed from NUM.")


if __name__ == "__main__":
    run_leakage_check()