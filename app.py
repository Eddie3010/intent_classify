from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import joblib
import numpy as np

# 1. Setup FastAPI and Schema
app = FastAPI(title="AI Sales Assistant Intent API", version="1.0")

class IntentRequest(BaseModel):
    previous_Intent: str
    previous_bot_response: str
    current_query: str

class IntentResponse(BaseModel):
    current_intent: str
    confidence_score: float

# 2. Define Model Structure again or import
class IntentClassifierMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(IntentClassifierMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# 3. Load Models on startup
preprocessors = None
model = None

@app.on_event("startup")
def load_model():
    global preprocessors, model
    try:
        preprocessors = joblib.load("preprocessors.pkl")
        model = IntentClassifierMLP(preprocessors['input_size'], preprocessors['num_classes'])
        model.load_state_dict(torch.load("model.pth", weights_only=True))
        model.eval()
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

# 4. Predict Endpoint
@app.post("/predict", response_model=IntentResponse)
def predict_intent(request: IntentRequest):
    if preprocessors is None or model is None:
        raise HTTPException(status_code=500, detail="Models are not loaded.")

    try:
        # Extract features
        prev_intent = request.previous_Intent
        bot_resp = request.previous_bot_response
        query = request.current_query

        # Preprocess
        # ohe expects 2D array, e.g., [['product_inventory']]
        X_prev = preprocessors['ohe'].transform([[prev_intent]])
        X_bot = preprocessors['tfidf_bot'].transform([bot_resp]).toarray()
        X_query = preprocessors['tfidf_query'].transform([query]).toarray()

        # Combine
        X_combined = np.hstack((X_prev, X_bot, X_query))
        X_tensor = torch.tensor(X_combined, dtype=torch.float32)

        # Inference
        with torch.no_grad():
            outputs = model(X_tensor)
            # Apply softmax to get confidence scores
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            confidence_score, predicted_class_index = torch.max(probabilities, 1)
            
            # Map back to original intent label
            predicted_intent = preprocessors['label_encoder'].inverse_transform(predicted_class_index.numpy())[0]

        return IntentResponse(
            current_intent=str(predicted_intent),
            confidence_score=float(confidence_score.item())
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
