import pandas as pd

file_path = "/home/kali/Downloads/AppTraffic.csv" # <-- Make sure this path is correct
df = pd.read_csv(file_path)

print("Columns in AppTraffic.csv:")
for col in df.columns:
    print(f"- '{col}'")

# You can also print the first few rows to eyeball the data
print("\nFirst 5 rows of AppTraffic.csv:")
print(df.head())
