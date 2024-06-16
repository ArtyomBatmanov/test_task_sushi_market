import pandas as pd


def generate_predictions(model, unique_dishes, df, next_day):
    future_predictions = []
    for dish in unique_dishes:
        average_discount_sum = df[df['DishName'] == dish]['DishDiscountSumInt'].mean()
        average_day_of_week = next_day.dayofweek
        average_hour = df[df['DishName'] == dish]['hour'].mean()

        future_data = pd.DataFrame({
            'DishDiscountSumInt': [average_discount_sum],
            'day_of_week': [average_day_of_week],
            'hour': [average_hour]
        })

        predicted_amount = model.predict(future_data)[0]
        predicted_amount = max(predicted_amount, 0)
        future_predictions.append([dish, predicted_amount])

    future_predictions_df = pd.DataFrame(future_predictions, columns=['DishName', 'PredictedAmount'])
    return future_predictions_df
