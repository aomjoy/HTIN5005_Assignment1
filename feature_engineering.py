import pandas as pd

print("loading filled_data.csv...")
full_df = pd.read_csv("filled_data.csv")
print(f"Loaded shape: {full_df.shape}")

# ---------- qSOFA Score ----------
print("Calculating qSOFA score...")
full_df["qSOFA"] = (
    (full_df["SBP"] < 100).astype(int) +
    (full_df["Resp"] >= 20).astype(int)
)

# ---------- NEWS Score ----------
def news_resp(x):
    if pd.isna(x): return 0
    if x <= 8: return 3
    elif x <= 11: return 1
    elif x <= 20: return 0
    elif x <= 24: return 2
    else: return 3

def news_o2sat(x):
    if pd.isna(x): return 0
    if x <= 91: return 3
    elif x <= 93: return 2
    elif x <= 95: return 1
    else: return 0

def news_sbp(x):
    if pd.isna(x): return 0
    if x <= 90: return 3
    elif x <= 100: return 2
    elif x <= 110: return 1
    elif x <= 219: return 0
    else: return 3

def news_hr(x):
    if pd.isna(x): return 0
    if x <= 40: return 3
    elif x <= 50: return 1
    elif x <= 90: return 0
    elif x <= 110: return 1
    elif x <= 130: return 2
    else: return 3

def news_temp(x):
    if pd.isna(x): return 0
    if x <= 35.0: return 3
    elif x <= 36.0: return 1
    elif x <= 38.0: return 0
    elif x <= 39.0: return 1
    else: return 2

print("Calculating NEWS score...")
full_df["NEWS"] = (
    full_df["Resp"].apply(news_resp) +
    full_df["O2Sat"].apply(news_o2sat) +
    full_df["SBP"].apply(news_sbp) +
    full_df["HR"].apply(news_hr) +
    full_df["Temp"].apply(news_temp)
)

# Vital signs we'll engineer extra features from
vital_cols = ["HR", "O2Sat", "Temp", "SBP", "Resp"]

print("Calculating lag and expanding window features (this may take a few minutes)...")
for col in vital_cols:
    print(f"  Processing {col}...")
    full_df[f"{col}_lag_diff"] = full_df.groupby("patient_id")[col].diff()
    full_df[f"{col}_expandingmean"] = (
        full_df.groupby("patient_id")[col].expanding().mean().reset_index(level=0, drop=True)
    )
    full_df[f"{col}_expandingmax"] = (
        full_df.groupby("patient_id")[col].expanding().max().reset_index(level=0, drop=True)
    )
    full_df[f"{col}_expandingmin"] = (
        full_df.groupby("patient_id")[col].expanding().min().reset_index(level=0, drop=True)
    )

print(f"Final shape after feature engineering: {full_df.shape}")

# ===== IMPORTANT: this was missing before =====
full_df.to_csv("featured_data.csv", index=False)
print("Saved featured_data.csv successfully!")