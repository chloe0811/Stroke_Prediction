def get_clean_stroke_data(local_save=True):
    import pandas as pd
    import os
    import kagglehub

    # Download data
    path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
    stroke_data = os.path.join(path, "healthcare-dataset-stroke-data.csv")
    df = pd.read_csv(stroke_data)

    # Fill missing values
    df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

    # Optional: Save locally
    if local_save:
        output_path = "/Users/sptsai/Documents/GitHub/Stroke_Prediction/stroke_data.csv"
        df.to_csv(output_path, index=False)
    
    return df
