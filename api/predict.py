import json
import os
import numpy as np
import xgboost as xgb
from http.server import BaseHTTPRequestHandler

BASE = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE, 'train', 'saved_models')

CLASS_NAMES = ['Out', 'Single', 'Double', 'Triple', 'HR']

# Module-level cache so models survive warm invocations
_cache = {}

def get_models():
    if _cache:
        return _cache['hit_features'], _cache['bases_features'], _cache['hit_model'], _cache['bases_model']

    # Feature names (must match what you trained with)
    hit_features   = ['hc_x', 'hc_y', 'launch_speed', 'launch_angle']
    bases_features = ['hc_x', 'hc_y', 'launch_speed', 'launch_angle']

    hit_model = xgb.XGBClassifier()
    hit_model.load_model(os.path.join(MODELS_DIR, 'hit_xgb.json'))

    bases_model = xgb.XGBClassifier()
    bases_model.load_model(os.path.join(MODELS_DIR, 'bases_xgb.json'))

    _cache['hit_features']   = hit_features
    _cache['bases_features'] = bases_features
    _cache['hit_model']      = hit_model
    _cache['bases_model']    = bases_model

    return hit_features, bases_features, hit_model, bases_model


class handler(BaseHTTPRequestHandler):

    def do_OPTIONS(self):
        self._cors()
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            body   = json.loads(self.rfile.read(length))

            hit_features, bases_features, hit_model, bases_model = get_models()

            # Validate all required fields are present
            for f in hit_features:
                if f not in body:
                    raise ValueError(f"Missing field: {f}")

            hit_input = np.array([[body[f] for f in hit_features]], dtype=np.float32)
            hit_prob  = float(hit_model.get_booster().predict(xgb.DMatrix(hit_input))[0])

            bases_input = np.array([[body[f] for f in bases_features]], dtype=np.float32)
            bases_probs = (
                bases_model.get_booster()
                .predict(xgb.DMatrix(bases_input))
                .reshape(1, -1)[0]
                .tolist()
            )

            # Normalise hit-type breakdown to sum to 100%
            bases_breakdown = {CLASS_NAMES[i]: bases_probs[i] for i in range(1, 5)}
            total           = sum(bases_breakdown.values())
            bases_breakdown = {k: round(v / total * 100, 1) for k, v in bases_breakdown.items()}

            self._send(200, {
                'hit_probability': round(hit_prob * 100, 1),
                'out_probability': round((1 - hit_prob) * 100, 1),
                'bases_breakdown': bases_breakdown,
            })

        except (KeyError, ValueError) as e:
            self._send(400, {'error': str(e)})
        except Exception as e:
            self._send(500, {'error': 'Internal server error'})

    # ── helpers ──────────────────────────────────────────────────────────────

    def _cors(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin',  '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type',                 'application/json')
        self.send_header('Content-Length',               str(len(body)))
        self.send_header('Access-Control-Allow-Origin',  '*')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # suppress default Apache-style request logging