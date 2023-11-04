import joblib
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import xgboost as xgb
import json

# Open model from pickle file
with open('xgboost_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_ = request.json
        print('json: ', json_)

        encoder = joblib.load('encoder.pkl')

        input_data = json.loads(json_)

        query = pd.DataFrame([input_data])

        query_encoded = encoder.transform(query)

        prediction = model.predict_proba(query_encoded).T[1].tolist()

        # Use low threshold to avoid false negatives
        return jsonify({'prediction': prediction, 'isFraud': prediction[0] > 0.2})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=False)
