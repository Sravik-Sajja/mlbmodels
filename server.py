from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import xgboost as xgb
import numpy as np
import requests
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

MLB_API      = 'https://statsapi.mlb.com/api/v1'
MLB_API_LIVE = 'https://statsapi.mlb.com/api/v1.1'


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


@app.route('/games', methods=['GET'])
@limiter.limit("60 per minute")
def get_games_for_date():
    date = request.args.get('date')
    if not date:
        return jsonify({'error': 'date query param is required (YYYY-MM-DD)'}), 400

    try:
        resp = requests.get(
            f"{MLB_API}/schedule",
            params={'sportId': 1, 'date': date},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        games = []
        for d in data.get('dates', []):
            for g in d.get('games', []):
                away = g.get('teams', {}).get('away', {})
                home = g.get('teams', {}).get('home', {})
                games.append({
                    'gamePk': g.get('gamePk'),
                    'status': g.get('status', {}).get('detailedState'),
                    'away': away.get('team', {}).get('name'),
                    'home': home.get('team', {}).get('name'),
                    'awayScore': away.get('score'),
                    'homeScore': home.get('score'),
                })

        return jsonify({'games': games})
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Could not reach MLB Stats API: {e}'}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/plays', methods=['GET'])
@limiter.limit("60 per minute")
def get_plays_for_game():
    game_pk = request.args.get('gamePk')
    if not game_pk:
        return jsonify({'error': 'gamePk query param is required'}), 400

    try:
        resp = requests.get(f"{MLB_API_LIVE}/game/{game_pk}/feed/live", timeout=10)
        resp.raise_for_status()
        data = resp.json()

        all_plays = data.get('liveData', {}).get('plays', {}).get('allPlays', [])
        plays = []

        for play in all_plays:
            batter = play.get('matchup', {}).get('batter', {}).get('fullName', 'Unknown')
            result = play.get('result', {})
            about = play.get('about', {})

            for pe in play.get('playEvents', []):
                hit_data = pe.get('hitData')
                if not hit_data:
                    continue
                coords = hit_data.get('coordinates') or {}
                if 'coordX' not in coords or 'coordY' not in coords:
                    continue
                if hit_data.get('launchSpeed') is None or hit_data.get('launchAngle') is None:
                    continue

                plays.append({
                    'batter': batter,
                    'event': result.get('event', ''),
                    'description': result.get('description', ''),
                    'inning': about.get('inning'),
                    'half': about.get('halfInning'),
                    'launch_speed': round(hit_data['launchSpeed'], 1),
                    'launch_angle': round(hit_data['launchAngle'], 1),
                    'distance': hit_data.get('totalDistance'),
                    'trajectory': hit_data.get('trajectory'),
                    'hc_x': round(coords['coordX'], 1),
                    'hc_y': round(coords['coordY'], 1),
                })

        return jsonify({'plays': plays})
    except requests.exceptions.RequestException as e:
        return jsonify({'error': f'Could not reach MLB Stats API: {e}'}), 502
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)