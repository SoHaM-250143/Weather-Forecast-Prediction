from flask import Flask, render_template, request, jsonify
import sqlite3
import requests
import matplotlib.pyplot as plt
import io, base64
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Replace with your own API key from OpenWeatherMap
API_KEY = "0a59c250820a6cbb1d281e9307eef38e"

# -------------------- DB Init --------------------
def init_db():
    conn = sqlite3.connect("weather.db")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS weather_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT,
            temperature REAL,
            humidity REAL,
            wind REAL,
            date_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_db()

# -------------------- Routes --------------------
@app.route("/")
def index():
    return render_template("index.html")

# Fetch Weather and Save
@app.route("/get_weather")
def get_weather():
    city = request.args.get("city")
    if not city:
        return jsonify({"error": "City is required"}), 400

    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if data.get("cod") != 200:
            return jsonify({"error": data.get("message", "API error")}), 400

        temperature = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind = data["wind"]["speed"]

        conn = sqlite3.connect("weather.db")
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO weather_log (city, temperature, humidity, wind) VALUES (?, ?, ?, ?)",
            (city, temperature, humidity, wind)
        )
        conn.commit()
        conn.close()

        return jsonify({
            "city": city,
            "temperature": temperature,
            "humidity": humidity,
            "wind": wind
        })

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# -------------------- Analysis Page --------------------
@app.route("/analysis")
def analysis():
    try:
        conn = sqlite3.connect("weather.db")
        cur = conn.cursor()
        cur.execute("SELECT city, temperature, humidity, wind, date_time FROM weather_log ORDER BY date_time DESC LIMIT 10")
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return "<h2>No data available for analysis yet. Please fetch some weather data first.</h2>"

        cities = [r[0] for r in rows]
        temps = [r[1] for r in rows]
        hums = [r[2] for r in rows]
        winds = [r[3] for r in rows]

        plt.figure(figsize=(8, 5))
        plt.plot(cities, temps, marker="o", label="Temperature (°C)")
        plt.plot(cities, hums, marker="s", label="Humidity (%)")
        plt.plot(cities, winds, marker="^", label="Wind Speed (m/s)")
        plt.legend()
        plt.title("Weather Analysis (Last 10 Records)")
        plt.xlabel("City")
        plt.xticks(rotation=45)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        buf.close()
        plt.close()

        return f"""
        <html>
            <head>
                <title>Weather Analysis</title>
                <style>
                    body {{
                        text-align: center;
                        font-family: Arial, sans-serif;
                        background: linear-gradient(135deg, #74ebd5, #ACB6E5);
                        margin: 0;
                        padding: 40px;
                    }}
                    h2 {{
                        color: #333;
                    }}
                    .btn {{
                        display: inline-block;
                        margin-top: 20px;
                        padding: 12px 25px;
                        border: none;
                        border-radius: 8px;
                        background: #4CAF50;
                        color: white;
                        font-size: 16px;
                        cursor: pointer;
                        text-decoration: none;
                        transition: transform 0.2s ease, background 0.3s ease;
                    }}
                    .btn:hover {{
                        background: #45a049;
                        transform: scale(1.05);
                    }}
                    img {{
                        border-radius: 12px;
                        box-shadow: 0 6px 15px rgba(0,0,0,0.2);
                        margin-top: 20px;
                        max-width: 90%;
                    }}
                </style>
            </head>
            <body>
                <h2>📊 Weather Analysis</h2>
                <img src="data:image/png;base64,{img_base64}" alt="Weather Analysis">
                <br>
                <a href="/" class="btn">🏠 Home</a>
            </body>
        </html>
        """

    except Exception as e:
        return f"<h2 style='color:red;'>Error generating analysis: {str(e)}</h2>"

# -------------------- Prediction --------------------
@app.route("/predict")
def predict():
    city = request.args.get("city")
    if not city:
        return jsonify({"error": "City is required"}), 400

    conn = sqlite3.connect("weather.db")
    cur = conn.cursor()
    cur.execute("SELECT id, temperature FROM weather_log WHERE city = ? ORDER BY id", (city,))
    rows = cur.fetchall()
    conn.close()

    if len(rows) < 3:
        return jsonify({"error": "Not enough data for prediction. Fetch weather at least 3 times."})

    X = np.array([r[0] for r in rows]).reshape(-1, 1)
    y = np.array([r[1] for r in rows])

    model = LinearRegression()
    model.fit(X, y)

    next_id = rows[-1][0] + 1
    predicted_temp = model.predict([[next_id]])[0]

    return jsonify({
        "city": city,
        "predicted_temperature": round(predicted_temp, 2),
        "data_points_used": len(rows)
    })

# -------------------- Run --------------------
if __name__ == "__main__":
    app.run(debug=True)
