import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib  # for saving trained models

# ===== Load train/test data saved from previous step =====
print("Loading train/test data...")
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv")["SepsisLabel"]
y_test = pd.read_csv("y_test.csv")["SepsisLabel"]

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# ===== Step 8: Train models =====
print("Training Random Forest...")
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

print("Training XGBoost...")
xgb_model = XGBClassifier(random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)

# ===== Step 9: Evaluate and compare with paper's numbers =====
print("\n===== Evaluation Results =====")
for name, model in [("Random Fo rest", rf_model), ("XGBoost", xgb_model)]:
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    auroc = roc_auc_score(y_test, probs)
    
    print(f"\n{name}:")
    print(f"  Accuracy = {acc:.4f}")
    print(f"  AUROC    = {auroc:.4f}")
    print(classification_report(y_test, preds))

print("\n===== Comparison with original paper (Mahmud et al. 2026, Table 7) =====")
print("Paper Random Forest: Accuracy = 86.49%, AUROC = 0.94")

# ===== Save trained models for the next step (SHAP) =====
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(xgb_model, "xgb_model.pkl")
print("\nSaved rf_model.pkl and xgb_model.pkl successfully!")