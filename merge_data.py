import pandas as pd
import os

# Path to the folder containing all downloaded .psv files
folder = "training_setA"
all_data = []

# List all .psv files in the folder
files = [f for f in os.listdir(folder) if f.endswith(".psv")]
print(f"Found {len(files)} files")

for i, filename in enumerate(files):
    filepath = os.path.join(folder, filename)
    df = pd.read_csv(filepath, sep="|")  # PhysioNet files are pipe-separated
    
    # Tag each row with its patient ID
    patient_id = filename.replace(".psv", "")
    df["patient_id"] = patient_id
    
    all_data.append(df)
    
    # Print progress every 2000 files (since there are tens of thousands of files)
    if (i + 1) % 2000 == 0:
        print(f"Merged {i + 1}/{len(files)} files")

print("Combining all files into one table...")
full_df = pd.concat(all_data, ignore_index=True)

print(f"Merged table shape: {full_df.shape}")
print(f"Total number of patients: {full_df['patient_id'].nunique()}")
print(f"Total sepsis-positive rows (label=1): {full_df['SepsisLabel'].sum()}")

# Save to a single file so we don't need to re-merge every time
full_df.to_csv("merged_data.csv", index=False)
print("Saved merged_data.csv successfully!")