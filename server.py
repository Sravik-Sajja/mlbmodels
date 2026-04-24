from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import xgboost as xgb
import numpy as np
import os

app = Flask(__name__)
CORS(app)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

hit_features   = ['hc_x', 'hc_y', 'launch_speed', 'launch_angle']
bases_features = ['hc_x', 'hc_y', 'launch_speed', 'launch_angle']

hit_model = xgb.XGBClassifier()
hit_model.load_model('train/saved_models/hit_xgb.json')

bases_model = xgb.XGBClassifier()
bases_model.load_model('train/saved_models/bases_xgb.json')

CLASS_NAMES = ['Out', 'Single', 'Double', 'Triple', 'HR']


@app.route('/predict', methods=['POST'])
@limiter.limit("30 per minute")
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)