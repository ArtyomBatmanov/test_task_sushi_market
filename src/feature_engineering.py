def create_features(df):
    df['day_of_week'] = df['CloseTime'].dt.dayofweek
    df['hour'] = df['CloseTime'].dt.hour
    return df