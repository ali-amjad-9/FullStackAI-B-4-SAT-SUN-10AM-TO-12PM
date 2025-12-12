from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the model
with open("C:\\Users\\HP\\Documents\\GitHub\\FULLSTACK-WITH-AI-BOOTCAMP-B1-MonToFri-2.5Month-Explorer\\predictive_maintenance_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse the input data (sensor readings)
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)

        # Make predictions
        prediction = model.predict(features)
        #result = "Failure predicted" if prediction[0] == 1 else "No failure predicted"
        print(prediction)
        return jsonify({"prediction": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)