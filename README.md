### Put your saved model + scaler + encoder into backend/saved_models/:
```
backend/saved_models/mlp_multiclass_model.keras
backend/saved_models/scaler.pkl
backend/saved_models/label_encoder.pkl
```

### Install and run:
```
cd backend
python3 -m venv venv
.\venv\Scripts\Activate.ps1   
pip install -r requirements.txt
python app.py
```

### The app will start on http://0.0.0.0:5000/ — open that in your browser.

### Upload a CSV with flow rows or paste a JSON object and click predict.