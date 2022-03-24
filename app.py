import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import contextvars

app = Flask(__name__)
model = pickle.load(open('breast_cancer_detector.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = [ 'mean concavity',
       'mean concave points', 'area error',
        'compactness error', 'concavity error',
       'worst texture',  'worst area',
       'worst smoothness', 
       'worst concave points', 'worst symmetry']
    
    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 0:
        res_val = "** Breast Cancer **"
    else:
        res_val = "No Breast Cancer"
        

    return render_template('index.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
    app.run(debug=True)
