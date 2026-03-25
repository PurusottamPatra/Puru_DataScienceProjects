from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__, template_folder="templates")

# Load model
try:
    with open("transaction_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("MODEL LOADED SUCCESSFULLY")
except Exception as e:
    print("Failed to load model:", e)
    model = None

# Categories for one-hot encoding
type_categories = ["Payment","Transfer","CashOut","Debit","Credit","CashIn","Refund"]
gender_categories = ["Male","Female"]

@app.route("/")
def home():
    return "<h2>Flask Backend Running</h2><a href='/predict-page'>Go to Prediction Page</a>"

@app.route("/predict-page")
def predict_page():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"status":"error","error":"Model not loaded"}), 500
    try:
        data = request.json

        # Convert incoming JSON to DataFrame
        df = pd.DataFrame([data])

        # Ensure categorical columns are strings
        df["Type"] = df["Type"].astype(str)
        df["Gender"] = df["Gender"].astype(str)

        # One-hot encode Type
        for t in type_categories:
            df[f"Type_{t}"] = (df["Type"] == t).astype(int)
        df.drop("Type", axis=1, inplace=True)

        # One-hot encode Gender
        for g in gender_categories:
            df[f"Gender_{g}"] = (df["Gender"] == g).astype(int)
        df.drop("Gender", axis=1, inplace=True)

        # Numeric columns
        for col in ["Amount","OldBalance","NewBalance","Age"]:
            df[col] = pd.to_numeric(df[col])

        # Align columns with model
        if hasattr(model, 'n_features_in_'):
            expected_cols = model.n_features_in_
            # Fill missing columns with 0
            for i, col in enumerate(expected_cols):
                if col not in df.columns:
                    df[col] = 0
            # Order columns correctly
            df = df[expected_cols]

        # Predict
        prediction = model.predict(df.values)[0]

        return jsonify({"status":"success","prediction":float(prediction)})

    except Exception as e:
        return jsonify({"status":"error","error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
