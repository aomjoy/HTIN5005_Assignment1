import pandas as pd

print("Loading merged_data.csv...")
full_df = pd.read_csv("merged_data.csv")

print(f"Shape before forward fill: {full_df.shape}")
print(f"Total missing values before: {full_df.isnull().sum().sum()}")

print("Applying forward fill per patient...")
# Apply forward fill separately for each patient (so data doesn't leak between patients)
full_df = full_df.groupby("patient_id", group_keys=False).apply(lambda x: x.ffill())

print(f"Total missing values after: {full_df.isnull().sum().sum()}")

# Save the filled data so we don't need to redo this step every time
full_df.to_csv("filled_data.csv", index=False)
print("Saved filled_data.csv successfully!")