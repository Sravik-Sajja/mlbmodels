from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import xgboost as xgb
import torch
import numpy as np

app = Flask(__name__)
CORS(app)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

# Load feature names from .pth checkpoints
hit_ckpt = torch.load('train/saved_models/hit_model.pth', map_location='cpu', weights_only=False)
hit_features = hit_ckpt['feature_names']

bases_ckpt = torch.load('train/saved_models/bases_model.pth', map_location='cpu', weights_only=False)
bases_features = bases_ckpt['feature_names']

# Load XGBoost models
hit_model = xgb.XGBClassifier()
hit_model.load_model('train/saved_models/hit_xgb.json')

bases_model = xgb.XGBClassifier()
bases_model.load_model('train/saved_models/bases_xgb.json')

CLASS_NAMES = ['Out', 'Single', 'Double', 'Triple', 'HR']


@app.route('/predict', methods=['POST'])
@limiter.limit("15 per minute")
def predict():
    data = request.json
    try:
        hit_input = np.array([[data[f] for f in hit_features]], dtype=np.float32)
        hit_prob = float(hit_model.get_booster().predict(xgb.DMatrix(hit_input))[0])

        bases_input = np.array([[data[f] for f in bases_features]], dtype=np.float32)
        bases_probs = bases_model.get_booster().predict(xgb.DMatrix(bases_input)).reshape(1, -1)[0].tolist()

        bases_breakdown = {CLASS_NAMES[i]: round(bases_probs[i] * 100, 1) for i in range(1, 5)}
        total = sum(bases_breakdown.values())
        bases_breakdown = {k: round(v / total * 100, 1) for k, v in bases_breakdown.items()}

        return jsonify({
            'hit_probability': round(hit_prob * 100, 1),
            'out_probability': round((1 - hit_prob) * 100, 1),
            'bases_breakdown': bases_breakdown,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(port=5000, debug=True)