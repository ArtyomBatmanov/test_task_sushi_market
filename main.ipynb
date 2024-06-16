import pandas as pd
from src.data_preprocessing import load_and_preprocess_data
from src.feature_engineering import create_features
from src.model_training import train_model
from src.prediction import generate_predictions

file_path = "data/Екатеринбург,_Гринвич,_ул_8_марта,_46.xlsx"
df = load_and_preprocess_data(file_path)

df = create_features(df)

X = df[["DishDiscountSumInt", "day_of_week", "hour"]]
y = df["DishAmountInt"]

model = train_model(X, y)

unique_dishes = df["DishName"].unique()
next_day = pd.to_datetime("2024-06-17")
predictions = generate_predictions(model, unique_dishes, df, next_day)

output_file = "predicted_sales_2024-06-17.xlsx"
predictions.to_excel(output_file, index=False)
print(f"Predictions saved to {output_file}")
