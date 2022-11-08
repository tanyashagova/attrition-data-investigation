import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model.bin'
# loading the model
with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('predict')


@app.route('/predict', methods=['POST'])
def predict():
    person = request.get_json()

    X = dv.transform([person])
    y_pred = model.predict_proba(X)[0, 1]
    attrit = y_pred >= 0.5

    result = {
        "attrition_probability": float(y_pred),
        "attrit": bool(attrit)
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
    