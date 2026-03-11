import numpy as np
import pickle
import os
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load model + scalers once at startup
model         = tf.keras.models.load_model('energy_model.keras')
feature_scaler= pickle.load(open('feature_scaler.pkl', 'rb'))
target_scaler = pickle.load(open('target_scaler.pkl', 'rb'))

LOOKBACK_STEPS = 90
FEATURE_COLUMNS = [
    'Aggregate', 'voltage', 'current',
    'hour', 'day_of_week', 'is_weekend', 'time_of_day',
    'power_rolling_mean_12', 'power_rolling_std_12',
    'power_rolling_mean_450',
    'power_lag_1', 'power_lag_12',
    'power_change', 'num_appliances_on', 'power_factor'
]

@app.route('/', methods=['GET'])
def home():
    return jsonify({'status': 'Energy Predictor API is live ✅'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Expect last 90 readings as list of dicts
        # [{'Aggregate': 1.2, 'voltage': 230, ...}, ...]
        readings = data['readings']

        if len(readings) < LOOKBACK_STEPS:
            return jsonify({
                'error': f'Need {LOOKBACK_STEPS} readings, got {len(readings)}'
            }), 400

        # Build feature array
        import pandas as pd
        df_input = pd.DataFrame(readings[-LOOKBACK_STEPS:])
        X = df_input[FEATURE_COLUMNS].values.astype(np.float32)

        # Scale
        X_scaled = feature_scaler.transform(X)
        X_scaled = X_scaled.reshape(1, LOOKBACK_STEPS, len(FEATURE_COLUMNS))

        # Predict
        y_scaled = model.predict(X_scaled, verbose=0)
        y_pred   = target_scaler.inverse_transform(y_scaled)[0]

        return jsonify({
            'status'          : 'success',
            'forecast_steps'  : len(y_pred),
            'forecast_minutes': len(y_pred) * 8 / 60,
            'predictions_watts': y_pred.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port)
