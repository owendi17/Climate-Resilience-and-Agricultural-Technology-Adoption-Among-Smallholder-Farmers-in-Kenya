"""
save_model.py
─────────────
Run this ONCE after executing Agri_Strengthened.ipynb to export
the trained pipeline so the Streamlit app can load it.

Usage:
    python save_model.py
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── 1. Load cleaned data ────────────────────────────────────────────────────
print("Loading cleaned data...")
try:
    df = pd.read_csv("cleaned_agriculture_data.csv")
    print(f"  Loaded {len(df):,} rows.")
except FileNotFoundError:
    raise FileNotFoundError(
        "cleaned_agriculture_data.csv not found.\n"
        "Run Agri_Strengthened.ipynb first to generate it."
    )

# ── 2. Define features and target ───────────────────────────────────────────
features = [
    'education level', 'gender', 'age group',
    'agricultural financing', 'phone ownership', 'internet use',
    'farming experience', 'climate_risk_score'
]
features = [f for f in features if f in df.columns]
target   = 'high_adoption'

if target not in df.columns:
    raise ValueError(f"'{target}' column not found. Run the notebook first.")

X = df[features].copy()
y = df[target].copy()

cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include='number').columns.tolist()

print(f"  Features: {features}")
print(f"  Target distribution: {y.value_counts().to_dict()}")

# ── 3. Preprocessor ─────────────────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
    ('num', StandardScaler(), num_cols)
], remainder='drop')

# ── 4. Candidate models ─────────────────────────────────────────────────────
models = {
    'Logistic Regression': Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ]),
    'Random Forest': Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=200, max_depth=6,
                                        min_samples_leaf=5, random_state=RANDOM_STATE))
    ]),
    'Gradient Boosting': Pipeline([
        ('pre', preprocessor),
        ('clf', GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                            learning_rate=0.05, random_state=RANDOM_STATE))
    ])
}

# ── 5. Cross-validate and pick best ─────────────────────────────────────────
print("\nCross-validating models (5-fold, ROC-AUC)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}

for name, pipe in models.items():
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    results[name] = scores.mean()
    print(f"  {name:25s}  AUC = {scores.mean():.3f} ± {scores.std():.3f}")

best_name = max(results, key=results.get)
print(f"\nBest model: {best_name} (AUC = {results[best_name]:.3f})")

# ── 6. Final train / test evaluation ────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

best_pipe = models[best_name]
best_pipe.fit(X_train, y_train)

y_pred  = best_pipe.predict(X_test)
y_proba = best_pipe.predict_proba(X_test)[:, 1]

print("\nTest set evaluation:")
print(classification_report(y_test, y_pred, target_names=['Low Adopter', 'High Adopter']))
print(f"ROC-AUC (test): {roc_auc_score(y_test, y_proba):.3f}")

# ── 7. Retrain on full dataset and save ─────────────────────────────────────
print("\nRetraining on full dataset...")
best_pipe.fit(X, y)

joblib.dump(best_pipe, "model_pipeline.pkl")
print("✓ Model saved to model_pipeline.pkl")
print("\nYou can now run:  streamlit run app.py")
