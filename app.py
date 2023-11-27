import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd 

app=Flask(__name__)
### Load the regression model
regmodel=pickle.load(open('regmodel.pkl','rb'))
std_scaler=pickle.load(open('scaler.pkl','rb'))



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=["POST"])

def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    std_data=std_scaler.transform(np.array(list(data.values())).reshape(1,-1))

    output=regmodel.predict(std_data)
    print(output[0])
    return jsonify(output[0])


if __name__ =='__main__':
    app.run(debug=True,port=8001)

