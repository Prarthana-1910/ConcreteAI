from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("models/Tuned_CatBoostRegressor.joblib") 

COSTS = np.array([6.0, 2.0, 3.6, 0.05, 1.1, 0.9, 150.0])
CO2_FACTORS = np.array([0.82, 0.02, 0.06, 0.0003, 0.005, 0.005, 0.02])

# ================= HOME PAGE =================
@app.route('/')
def home():
    return render_template("index.html")

# ================= PREDICTION ROUTE =================
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get JSON Data from the HTML Frontend
        data = request.get_json()

        # 2. Extract Variables
        cement = float(data['cement'])
        flyash = float(data['flyash'])
        ggbs = float(data['ggbs'])
        water = float(data['water'])
        coarse_agg = float(data['coarse_agg'])
        fine_agg = float(data['fine_agg'])  # Sand
        admixture = float(data['admixture'])
        age = float(data['days'])

        # 3. Feature Engineering: Calculate W/B Ratio
        # WBRatio = Water / (Cement + FlyAsh + GGBS)
        binder = cement + flyash + ggbs
        if binder == 0:
            wb_ratio = 0
        else:
            wb_ratio = water / binder

        # 4. Prepare Input Vector for Model
        # MUST Match the Training Order: 
        # ['Cement', 'GGBS', 'FlyAsh', 'Water', 'CoarseAggregate', 'Sand', 'Admixture', 'WBRatio', 'age']
        features = np.array([[cement, ggbs, flyash, water, coarse_agg, fine_agg, admixture, wb_ratio, age]])

        # 5. Predict Strength
        strength = model.predict(features)[0]

        # 6. Calculate Cost & CO2 (Raw Materials Only)
        # Order: Cement, FlyAsh, GGBS, Water, CoarseAgg, FineAgg, Admixture
        quantities = np.array([cement, flyash, ggbs, water, coarse_agg, fine_agg, admixture])
        
        cost = np.sum(quantities * COSTS)
        co2 = np.sum(quantities * CO2_FACTORS)

        # 7. Return JSON with simple keys matching your HTML JS
        return jsonify({
            "strength": round(strength, 2),
            "cost": round(cost, 2),
            "co2": round(co2, 2)
        })

    except Exception as e:
        print(f"Error: {e}") # Print error to console for debugging
        return jsonify({"error": str(e)}), 500

# ================= RUN SERVER =================
if __name__ == "__main__":
    app.run(debug=True)