
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("data/citynet_churn.csv")

# 2. Separate features and target
# X = df.drop(columns=["customer_id", "churned"])

X = df.drop(columns=[
    "customer_id",
    "churned",
    "cancel_request_logged"
])
y = df["churned"]

# 3. Identify numerical and categorical columns
numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# 4. Handle missing values and scale numerical data
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# 5. Handle categorical data
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# 6. Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# 7. Create SVC Linear model
model = Pipeline([
    ("preprocessor", preprocessor),
    # ("classifier", SVC(kernel="linear"))
    ("classifier", SVC(kernel="linear", class_weight="balanced"))
])

# 8. Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 9. Train the model
model.fit(X_train, y_train)

# 10. Make predictions
predictions = model.predict(X_test)

# 11. Evaluate model
print("Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))

# 12. Save trained model
joblib.dump(model, "models/svc_churn_model.joblib")

print("\nModel saved successfully!")



# 13. Load new customers
new_customers = pd.read_csv("data/new_customers.csv")

# 14. Remove customer ID from input features
new_X = new_customers.drop(columns=["customer_id"])

# 15. Predict churn
new_predictions = model.predict(new_X)

# 16. Calculate decision scores
scores = model.decision_function(new_X)

# 17. Create results
results = new_customers.copy()
results["predicted_churn"] = new_predictions
results["churn_score"] = scores

# 18. Sort customers by highest churn score
results = results.sort_values(
    by="churn_score", ascending=False
)

# 19. Select top 100 customers
top_100 = results.head(100)

# 20. Save report
top_100.to_csv("reports/top_100_customers.csv", index=False)

print("\nTop 100 customers:")
print(top_100.head(10))

print("\nTop 100 report saved successfully!")
