# Terminal - python app.py 
# Browser - http://127.0.0.1:5000/

from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load dataset (for locations + recommendations)
df = pd.read_csv("Cleaned_data.csv")
print("COLUMNS:", df.columns)
locations = sorted(df["location"].unique())

# Load the model
with open("RidgeModel.pkl", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html", locations=locations)


@app.route("/predict", methods=["POST"])
def predict():
    location = request.form.get("location")
    sqft = float(request.form.get("sqft"))
    bath = float(request.form.get("bath"))
    bhk = int(request.form.get("bhk"))

    # ---------- Prediction ----------
    try:
        # If model is a pipeline (handles encoding inside)
        data = pd.DataFrame([{
            "location": location,
            "total_sqft": sqft,
            "bath": bath,
            "bhk": bhk
        }])
        predicted = model.predict(data)[0]

    except Exception:
        # If model expects manual one-hot vector
        x = np.zeros(len(locations) + 3)
        x[0] = sqft
        x[1] = bath
        x[2] = bhk

        if location in locations:
            x[3 + locations.index(location)] = 1

        predicted = model.predict([x])[0]

    price_in_rupees = predicted * 100000


    # ---------- Recommendation System ----------
    df_rec = df.copy()
    df_rec["price_rupees"] = df_rec["price"] * 100000

    low = price_in_rupees - 2000000     # -20 lakhs
    high = price_in_rupees + 2000000    # +20 lakhs

    similar = df_rec[
        (df_rec["price_rupees"] >= low) &
        (df_rec["price_rupees"] <= high)
    ].copy()

    similar["distance"] = abs(similar["price_rupees"] - price_in_rupees)
    similar = similar.sort_values("distance").head(5)

    # Build final recommendations dictionary exactly matching HTML
    recommendations = []
    for _, row in similar.iterrows():
        recommendations.append({
            "location": row["location"],
            "sqft": row["total_sqft"],
            "bhk": row["bhk"],
            "bath": row["bath"],
            "price": row["price_rupees"]
        })

    print("RECOMMENDATIONS:", recommendations)
    # ---------- Render ----------
    return render_template(
        "result.html",
        price=round(price_in_rupees, 2),
        location=location,
        sqft=sqft,
        bath=bath,
        bhk=bhk,
        recommendations=recommendations
    )


if __name__ == "__main__":
    app.run(debug=True)




