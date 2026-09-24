import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import time

print("Loading model and test data...")
rf_model = joblib.load("rf_model.pkl")
X_test = pd.read_csv("X_test.csv")

X_test_sample = X_test.sample(n=1000, random_state=42)
print(f"Using a sample of {X_test_sample.shape[0]} rows for SHAP (faster)")

start_time = time.time()
print("Computing SHAP values...")
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test_sample, check_additivity=False)

elapsed = time.time() - start_time
print(f"SHAP computation took {elapsed/60:.2f} minutes")

# ===== IMPORTANT: check the shape first to know how to slice correctly =====
print(f"Type: {type(shap_values)}")
if isinstance(shap_values, list):
    print(f"List length: {len(shap_values)}, each array shape: {shap_values[0].shape}")
    shap_values_to_plot = shap_values[1]  # old-style: list of [class0, class1]
else:
    print(f"Array shape: {shap_values.shape}")
    if shap_values.ndim == 3:
        # new-style: (n_samples, n_features, n_classes) -> slice class 1 on the LAST axis
        shap_values_to_plot = shap_values[:, :, 1]
    else:
        shap_values_to_plot = shap_values

print(f"Final shape used for plotting: {shap_values_to_plot.shape}")

# ===== Plot summary (beeswarm) =====
print("Generating summary plot...")
shap.summary_plot(shap_values_to_plot, X_test_sample, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=150)
plt.close()
print("Saved shap_summary_plot.png")

# ===== Plot bar chart (global feature importance) =====
shap.summary_plot(shap_values_to_plot, X_test_sample, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_bar_plot.png", dpi=150)
plt.close()
print("Saved shap_bar_plot.png")

print("\nDone!")