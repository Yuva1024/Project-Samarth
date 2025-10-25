import pandas as pd
import os

print("🧹 CLEANING DATA\n")

DATA_FILES = [
    ("rainfall_subdivisional.csv", "data/rainfall_subdivisional_clean.csv"),
    ("wheat_punjab.csv", "data/wheat_punjab_clean.csv"),
    ("rice_karnataka.csv", "data/rice_karnataka_clean.csv"),
    ("crops_gujarat.csv", "data/crops_gujarat_clean.csv")
]

for infile, outfile in DATA_FILES:
    src = os.path.join('data', infile)
    if not os.path.exists(src):
        print(f"❌ {src} not found! Please check your data folder")
        continue

    df = pd.read_csv(src)
    print(f"✅ Loaded {infile}")

    # Standardize columns
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Remove nulls in key columns if possible
    maybe_year_cols = [col for col in df.columns if 'year' in col]
    maybe_state_cols = [col for col in df.columns if 'state' in col or 'district' in col]
    subset_cols = (maybe_state_cols + maybe_year_cols)[:2]
    if subset_cols:
        before = len(df)
        df = df.dropna(subset=subset_cols)
        print(f"   Removed {before - len(df)} rows with missing valuable data")

    print(f"   Columns: {list(df.columns)}")
    print(f"   Clean rows: {len(df)}")

    df.to_csv(outfile, index=False)
    print(f"✅ Saved cleaned CSV: {outfile}\n")

print("\n✅ DATA CLEANING COMPLETE")
