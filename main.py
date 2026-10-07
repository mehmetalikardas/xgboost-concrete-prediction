
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


# ============================================================
# 1. VERİ SETİNİ YÜKLE
# ============================================================

# Veri setini proje klasöründeki "data" klasörüne koyabilirsiniz.
data = pd.read_excel("data/Concrete_Data.xls")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# ============================================================
# 2. TRAIN / TEST AYRIMI
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 3. CROSS VALIDATION
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 4. XGBOOST MODELİ
# ============================================================

model = XGBRegressor(
    n_estimators=1000,
    random_state=42
)


# ============================================================
# 5. HYPERPARAMETER GRID
# ============================================================

param_grid = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.1]
}


# ============================================================
# 6. GRID SEARCH
# ============================================================

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
# 7. TEST TAHMİNLERİ
# ============================================================

y_pred = best_model.predict(X_test)


# ============================================================
# 8. TEST METRİKLERİ
# ============================================================

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# ============================================================
# 9. CROSS VALIDATION SONUÇLARI
# ============================================================

scores = cross_val_score(
    best_model,
    X_train,
    y_train,
    cv=kf,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

rmse_scores = -scores


# ============================================================
# 10. SONUÇLARI YAZDIR
# ============================================================

print("\n" + "=" * 50)
print("XGBOOST CONCRETE STRENGTH PREDICTION")
print("=" * 50)

print("\nCross Validation RMSE:")
print(rmse_scores)

print(f"\nMean CV RMSE: {rmse_scores.mean():.4f}")
print(f"CV RMSE Std:  {rmse_scores.std():.4f}")

print("\nTest Results:")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE:  {mae:.4f}")
print(f"Test R²:   {r2:.4f}")

print("\nBest Parameters:")
print(grid_search.best_params_)

print(f"\nBest Grid Search CV RMSE: {-grid_search.best_score_:.4f}")


# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

print("\nFeature Importance:")

feature_importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

for feature, importance in feature_importance.items():
    print(f"{feature}: {importance:.4f}")