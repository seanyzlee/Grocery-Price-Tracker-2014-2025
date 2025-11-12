import pandas as pd
import os, zipfile, re
import numpy as np

food_prices = pd.read_csv("food_prices.csv", low_memory=False, on_bad_lines='skip')

# Strip spaces and remove weird unnamed columns
food_prices.columns = food_prices.columns.str.strip()
food_prices = food_prices.loc[:, ~food_prices.columns.str.contains('^Unnamed', case=False, na=False)]

# Drop columns that are fully NaN or only contain blank/space values
food_prices = food_prices.dropna(axis=1, how='all')
food_prices = food_prices.loc[:, (food_prices.applymap(lambda x: str(x).strip() != '')).any()]

# Drop any column whose header looks like commas or garbage
food_prices = food_prices.loc[:, ~food_prices.columns.str.match(r'^[,.\s]*$')]

# Print summary
print(f"✅ Columns kept after cleaning: {len(food_prices.columns)}")
print("🧾 Sample columns:", food_prices.columns[:10].tolist())

template = pd.read_csv("products_beefstew.csv")


os.makedirs("product_files", exist_ok=True)

def clean_product_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r'\b(per|kilogram|gram|grams|litre|litres|millilitre|millilitres|unit|dozen)\b', '', name)
    name = re.sub(r'[^a-z]+', '', name)
    return name.strip()

product_columns = food_prices.columns[1:]  # skip first (time) column
for col in product_columns:
    clean_name = clean_product_name(col)
    if not clean_name:
        continue  # skip totally empty columns
    
    df = template.copy()
    df["Product"] = clean_name
    df["Price"] = food_prices[col].values[:len(df)]
    
    path = f"product_files/products_{clean_name}.csv"
    df.to_csv(path, index=False)
    print(f"✅ Created {path}")

zip_path = "products_all_clean.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
    for f in os.listdir("product_files"):
        zipf.write(os.path.join("product_files", f), f)
