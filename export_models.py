import pandas as pd

# Load the dataset
file_path = "datasets/raw_train.csv"
df = pd.read_csv(file_path)

# Select unique 'Model' along with 'Manufacturer'
df_unique = df[['Model', 'Manufacturer']].drop_duplicates(subset=['Model'])

# Save to a new CSV file
output_path = "datasets/Car_Models.csv"
df_unique.to_csv(output_path, index=False)

print(f"Unique car models saved to {output_path}")
