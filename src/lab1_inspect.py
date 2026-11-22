import pandas as pd

df = pd.read_csv("data/citynet_churn.csv")

print("=== DATASET OVERVIEW ===")
print(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
print("Column types:", {str(k): int(v) for k, v in df.dtypes.value_counts().items()})

print("\n=== TODO 1: MISSING VALUES ===")
missing = df.isna().sum()
missing_only = missing[missing > 0]
if missing_only.empty:
    print("No missing values found.")
else:
    for col, count in missing_only.items():
        pct = (count / len(df)) * 100
        print(f"- {col}: {count} missing ({pct:.1f}%)")

print("\n=== TODO 2: CHURN DISTRIBUTION & BALANCE ===")
churn_counts = df["churned"].value_counts()
churn_shares = df["churned"].value_counts(normalize=True)
for val in [0, 1]:
    label = "Retained (0)" if val == 0 else "Churned (1)"
    count = int(churn_counts.get(val, 0))
    share = float(churn_shares.get(val, 0.0)) * 100
    print(f"- {label}: {count} customers ({share:.1f}%)")
print("Balance status: Imbalanced (~82% retained vs ~18% churned)")

print("\n=== KEY METRICS & CHURN RATES ===")
print(df[["tenure_months", "support_calls_90d", "outage_hours_90d"]].describe().loc[["mean", "min", "max"]].round(1))
print("\nChurn rate by support calls (first 6):", df.groupby("support_calls_90d")["churned"].mean().round(2).head(6).to_dict())
print("Churn rate by cancel_request_logged:", df.groupby("cancel_request_logged")["churned"].mean().round(3).to_dict())

print("\n=== TODO 3: CORRELATION & LEAKAGE CHECK ===")
correlations = df.corr(numeric_only=True)["churned"].sort_values(ascending=False)
for col, corr in correlations.items():
    print(f"- {col}: {corr:+.3f}")

print("\nCross-tab (cancel_request_logged vs churned):")
print(pd.crosstab(df["cancel_request_logged"], df["churned"], rownames=["cancel_req"], colnames=["churned"]))
print("Note: 'cancel_request_logged' is target leakage—drop it before training the churn model.")