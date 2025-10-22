import pandas as pd
import os
import glob

folder = "product_files"

all_files = glob.glob(os.path.join(folder, "products_*.csv"))

dataframes = []
for file in all_files:
    try:
        df = pd.read_csv(file)
        dataframes.append(df)
        print(f"✅ Merged {os.path.basename(file)}")
    except Exception as e:
        print(f"⚠️ Skipped {file}: {e}")

merged = pd.concat(dataframes, ignore_index=True)
merged.to_csv("all_products_merged.csv", index=False)

print(f"\n🎉 Done! Combined {len(all_files)} files into all_products_merged.csv")
