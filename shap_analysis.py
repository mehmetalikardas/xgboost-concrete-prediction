
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
import shap


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_excel("data/Concrete_Data.xls")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 3. XGBOOST MODEL
# ============================================================

model = XGBRegressor(
    n_estimators=1000,
    random_state=42
)


# ============================================================
# 4. HYPERPARAMETER SEARCH
# ============================================================

param_grid = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.1]
}


grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_


# ============================================================
# 5. SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(best_model)

shap_values = explainer.shap_values(X_test)


# ============================================================
# 6. GLOBAL FEATURE IMPORTANCE
# ============================================================

print("\nFeature importance based on mean absolute SHAP values:\n")

shap_df = pd.DataFrame(
    shap_values,
    columns=X_test.columns
)

mean_shap = (
    shap_df.abs()
    .mean()
    .sort_values(ascending=False)
)

for feature, value in mean_shap.items():
    print(f"{feature}: {value:.4f}")


# ============================================================
# 7. SHAP SUMMARY PLOT
# ============================================================

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.show()


# ============================================================
# 8. AGE FEATURE ANALYSIS
# ============================================================

age_shap = shap_df["Age (day)"]

print("\nAge (day) SHAP values:")
print(f"Minimum: {age_shap.min():.4f}")
print(f"Maximum: {age_shap.max():.4f}")
print(f"Mean:    {age_shap.mean():.4f}")


# ============================================================
# 9. AGE VS SHAP VALUE
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    X_test["Age (day)"],
    age_shap
)

plt.xlabel("Age (day)")
plt.ylabel("SHAP Value")
plt.title("Effect of Age on Model Predictions")
plt.axhline(0)

plt.tight_layout()
plt.show()
