# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Recevoir le DataFrame depuis l'application
        data = pd.read_json(request.get_json()['client_data'], orient='split')
        predictions = model.predict_proba(data)[:, 0]
        # Retourner les scores prédits
        return jsonify({'scores': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)