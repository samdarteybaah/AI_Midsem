from flask import Flask, render_template, request, jsonify
import joblib
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('/Users/pkatobrah/Desktop/Year 2 Sem 2/Intro to AI/AIMIDSEM/AImidsem.html')


'''with open('/Users/pkatobrah/Desktop/Year 2 Sem 2/Intro to AI/random_forest_model.pkl', 'rb') as f:
        model_rf = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
        data = request.json
        features = [data['potential'], data['mentality_composure'], data['dribbling'], data['passing'], data['value_eur'], data['wage_eur']]  
        prediction = model_rf.predict([features])
        return jsonify({'prediction': prediction[0]})'''

if __name__ == '__main__':
    app.run( debug = True)


