from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import torch
import numpy as np
from train.models import HitNN, BasesNN

app = Flask(__name__)
CORS(app)

# Load hit model
hit_ckpt = torch.load('train/saved_models/hit_model.pth', map_location='cpu', weights_only=False)
hit_scaler = hit_ckpt['scaler']
hit_features = hit_ckpt['feature_names']
hit_model = HitNN(n_input_features=hit_ckpt['n_input_features'])
hit_model.load_state_dict(hit_ckpt['model_state_dict'])
hit_model.eval()

# Load bases model
bases_ckpt = torch.load('train/saved_models/bases_model.pth', map_location='cpu', weights_only=False)
bases_scaler = bases_ckpt['scaler']
bases_features = bases_ckpt['feature_names']
n_classes = bases_ckpt['n_classes']
bases_model = BasesNN(n_input_features=bases_ckpt['n_input_features'], n_classes=n_classes)
bases_model.load_state_dict(bases_ckpt['model_state_dict'])
bases_model.eval()

CLASS_NAMES = {5: ['Out', 'Single', 'Double', 'Triple', 'HR']}


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        # Hit probability
        hit_input = np.array([[data[f] for f in hit_features]], dtype=np.float32)
        hit_scaled = hit_scaler.transform(hit_input)
        hit_tensor = torch.tensor(hit_scaled, dtype=torch.float32)

        with torch.no_grad():
            hit_logit = hit_model(hit_tensor)
            hit_prob = torch.sigmoid(hit_logit).item()

        # Bases breakdown
        bases_input = np.array([[data[f] for f in bases_features]], dtype=np.float32)
        bases_scaled = bases_scaler.transform(bases_input)
        bases_tensor = torch.tensor(bases_scaled, dtype=torch.float32)

        with torch.no_grad():
            bases_logits = bases_model(bases_tensor)
            bases_probs = torch.softmax(bases_logits, dim=1).squeeze().tolist()

        labels = CLASS_NAMES.get(n_classes, [str(i) for i in range(n_classes)])
        bases_breakdown = {labels[i]: round(bases_probs[i] * 100, 1) for i in range(1, n_classes)}

        total_percentage = sum(bases_breakdown.values())
        for label in bases_breakdown:
            bases_breakdown[label] = round(bases_breakdown[label] / total_percentage * 100, 1)

        return jsonify({
            'hit_probability': round(hit_prob * 100, 1),
            'out_probability': round((1 - hit_prob) * 100, 1),
            'bases_breakdown': bases_breakdown,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(port=5000, debug=True)