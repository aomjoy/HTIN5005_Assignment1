import pandas as pd
full_df = pd.read_csv("featured_data.csv")

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Separate features (X) from the target label (y)
X = full_df.drop(columns=["SepsisLabel", "patient_id"])
y = full_df["SepsisLabel"]

# Fill any remaining missing values (e.g., from lag/expanding features at the start of a patient's record)
X = X.fillna(0)

# Select the 50 most informative features based on ANOVA F-score
selector = SelectKBest(f_classif, k=50)
X_selected = selector.fit_transform(X, y)

selected_columns = X.columns[selector.get_support()]
print(selected_columns.tolist())

# Split data before oversampling, to avoid data leakage into the test set
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

# Apply SMOTE only to the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# ===== Save train/test data for the next step =====
print("Saving train/test data...")
pd.DataFrame(X_train_resampled, columns=selected_columns).to_csv("X_train.csv", index=False)
pd.DataFrame(X_test, columns=selected_columns).to_csv("X_test.csv", index=False)
pd.DataFrame(y_train_resampled, columns=["SepsisLabel"]).to_csv("y_train.csv", index=False)
pd.DataFrame(y_test, columns=["SepsisLabel"]).to_csv("y_test.csv", index=False)
print("Saved X_train.csv, X_test.csv, y_train.csv, y_test.csv successfully!")