from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

# Load the model
with open("C:\\Users\\HP\\Documents\\GitHub\\FULLSTACK-WITH-AI-BOOTCAMP-B1-MonToFri-2.5Month-Explorer\\predictive_maintenance_model.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Define input data schema
class Features(BaseModel):
    features: float

@app.post("/predict")
def predict(features: Features):
    try:
        # Convert features to a numpy array
        data = np.array(features.features).reshape(1, -1)

        # Make predictions
        
        prediction = model.predict(data)
        print(prediction)
        return {"prediction": float(prediction)}
    
       # result = "Failure predicted" if prediction[0] == 1 else "No failure predicted"
       # return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))