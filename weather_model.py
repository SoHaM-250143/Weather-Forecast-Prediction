import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression

def train_and_predict():
    conn = sqlite3.connect("weather.db")
    df = pd.read_sql_query("SELECT * FROM weather_log", conn)
    conn.close()

    if len(df) < 5:  # need enough data for ML
        return "Not enough data for prediction"

    # Use simple time index as feature
    df["id"] = range(len(df))
    X = df[["id"]]
    y = df["temperature"]

    model = LinearRegression()
    model.fit(X, y)

    next_day = [[len(df) + 1]]
    prediction = model.predict(next_day)[0]

    return f"Predicted next temperature: {prediction:.2f} °C"
