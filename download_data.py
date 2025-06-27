import pandas as pd
import os
import kagglehub

# download data
path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
print("Path to dataset files:", path)

stroke_data = os.path.join(path, "healthcare-dataset-stroke-data.csv")

# download csv
df = pd.read_csv(stroke_data)

# show first 5 data
print(df.head())

# store to local
output_path = "/Users/sptsai/Documents/GitHub/Stroke_Prediction/healthcare-dataset-stroke-data.csv"
df.to_csv(output_path, index=False)
print("CSV saved to:", output_path)