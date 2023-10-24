from flask import Flask, render_template, request, jsonify
import joblib
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


with open("/Users/pkatobrah/Desktop/Year 2 Sem 2/Intro to AI/AIMIDSEM/random_forest_model (1).pkl", 'rb') as f:
        model_rf = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def predict():
        if request.method == "POST":
                to_predict_list = request.form.to_dict()

                prediction = preprocessDataAndPredict(to_predict_list)    
        return render_template("index.html", result = prediction)

def preprocessDataAndPredict(feature_dict):
        
        test_data = {k:[v] for k,v in feature_dict.items()}
        test_data = pd.DataFrame(test_data)
        standardScaler = StandardScaler()
        scaled_test_data = standardScaler.fit_transform(test_data.copy())

        file = open("/Users/pkatobrah/Desktop/Year 2 Sem 2/Intro to AI/AIMIDSEM/random_forest_model (1).pkl","rb")

        trained_model = joblib.load(file)

        predict = trained_model.predict(scaled_test_data)

        return predict


if __name__ == '__main__':
    app.run(debug = True)
    

'''from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the pre-trained model using joblib
model_rf = joblib.load("/Users/pkatobrah/Desktop/Year 2 Sem 2/Intro to AI/AIMIDSEM/random_forest_model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if request.method == "POST":
        feature_dict = request.form.to_dict()

        prediction = preprocessDataAndPredict(feature_dict)
        return render_template("index.html", result=prediction)

def preprocessDataAndPredict(feature_dict):
    test_data = pd.DataFrame([feature_dict])
    scaler = StandardScaler()
    scaled_test_data = scaler.fit_transform(test_data)

    predict = model_rf.predict(scaled_test_data)
    
    return predict

if __name__ == '__main__':
    app.run(debug=True)
'''