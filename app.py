from flask import Flask, render_template, request
import pickle
import numpy as np
import datetime

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    N = float(request.form["N"])
    P = float(request.form["P"])
    K = float(request.form["K"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    features = [[N, P, K, temperature, humidity, ph, rainfall]]
    prediction = model.predict(features)[0]

    warnings = []
    if temperature > 35:
        warnings.append("🔴 High temperature. Use drip irrigation.")
    elif temperature < 10:
        warnings.append("🔴 Too cold. Consider greenhouse farming.")
    if rainfall < 30:
        warnings.append("🔴 Drought risk. Use water conservation.")
    elif rainfall > 250:
        warnings.append("🔴 Flood risk. Improve drainage.")
    if humidity > 90:
        warnings.append("🟠 Very high humidity. Watch for fungal diseases.")
    if ph < 5.5:
        warnings.append("🟠 Soil too acidic. Add lime.")
    elif ph > 7.5:
        warnings.append("🟠 Soil too alkaline. Add sulfur.")
    if not warnings:
        warnings.append("✅ All conditions look good for farming!")

    warning = warnings[0]

    month = datetime.datetime.now().month
    if month in [3,4,5]:
        season = "🌸 Spring"
    elif month in [6,7,8,9]:
        season = "☀️ Summer / Kharif"
    elif month in [10,11]:
        season = "🍂 Rabi Season"
    else:
        season = "❄️ Winter"

    kharif_crops = ['rice','maize','cotton','sugarcane','groundnut','jute']
    rabi_crops = ['wheat','barley','mustard','peas','lentil']

    if prediction.lower() in kharif_crops:
        season_match = "✅ Perfect season to grow " + prediction + "!"
    elif prediction.lower() in rabi_crops:
        season_match = "⚠️ Best grown in Rabi season. Consider waiting."
    else:
        season_match = "🌿 This crop grows in multiple seasons."

    return render_template("index.html",
                           prediction=prediction,
                           warning=warning,
                           warnings=warnings,
                           season=season,
                           season_match=season_match,
                           N=N, P=P, K=K,
                           temperature=temperature,
                           humidity=humidity,
                           ph=ph,
                           rainfall=rainfall)

if __name__ == "__main__":
    app.run(debug=True)