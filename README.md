# XGBoost Concrete Strength Prediction

An XGBoost-based machine learning project for predicting concrete compressive strength from concrete mixture properties.

## 📌 About the Project

This project uses **XGBoost Regression** to predict the compressive strength of concrete based on its mixture properties and age.

The project focuses on:

- XGBoost regression
- Hyperparameter optimization with GridSearchCV
- 5-Fold Cross Validation
- Model evaluation using RMSE, MAE, and R²
- Feature importance analysis

## 📊 Dataset

The project uses the **Concrete Compressive Strength Dataset**.

The dataset contains the following features:

- Cement
- Blast Furnace Slag
- Fly Ash
- Water
- Superplasticizer
- Coarse Aggregate
- Fine Aggregate
- Age

The target variable is:

- **Concrete Compressive Strength (MPa)**

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- SHAP

## ⚙️ Methodology

The project follows these steps:

1. Load and preprocess the dataset
2. Split the data into training and test sets
3. Create an XGBoost regression model
4. Perform hyperparameter optimization using `GridSearchCV`
5. Evaluate the optimized model on the test set
6. Perform 5-Fold Cross Validation
7. Analyze feature importance
8. Use SHAP for model interpretability

## 🔍 Hyperparameter Optimization

`GridSearchCV` was used to search for the best combination of:

- `max_depth`
- `learning_rate`

The model was evaluated using **Root Mean Squared Error (RMSE)**.

## 📈 Model Evaluation

The model was evaluated using three main metrics:

- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **R²** — Coefficient of Determination

### Test Results

| Metric | Score |
|---|---:|
| RMSE | 4.17 MPa |
| MAE | 2.86 MPa |
| R² | 0.93 |

### 5-Fold Cross Validation

The model was evaluated using 5-Fold Cross Validation.

- Mean CV RMSE: **4.74 MPa**
- Standard Deviation: **0.45 MPa**

The relatively low standard deviation indicates that the model's performance was reasonably consistent across different validation folds.

## ⭐ Feature Importance

The most important features according to the XGBoost model were:

1. Age
2. Cement
3. Water
4. Blast Furnace Slag
5. Superplasticizer
6. Fine Aggregate
7. Coarse Aggregate
8. Fly Ash

## 🧠 Model Interpretability

SHAP was used to investigate how individual features influence the model's predictions.

This allows the model to be interpreted beyond simply looking at its prediction accuracy.

SHAP analysis helps answer questions such as:

- Which features have the greatest influence on predictions?
- Does increasing a feature increase or decrease the predicted strength?
- How does a specific feature affect individual predictions?

## 📁 Project Structure

```text
xgboost-concrete-prediction/
│
├── main.py
├── shap_analysis.py
├── requirements.txt
├── .gitignore
├── README.md
│
└── data/
    └── README.md
```

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/mehmetalikardas/xgboost-concrete-prediction.git
```

Navigate to the project directory:

```bash
cd xgboost-concrete-prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## ▶️ Usage

Run the main model:

```bash
python main.py
```

For SHAP analysis:

```bash
python shap_analysis.py
```

## 📌 Future Improvements

Possible future improvements include:

- More extensive hyperparameter optimization
- Additional regression models for comparison
- Advanced feature engineering
- Improved SHAP visualizations
- Deployment as a web application or API

## 👨‍💻 Author

Developed as a machine learning project for exploring **XGBoost regression, model optimization, cross-validation, and explainable AI**.
