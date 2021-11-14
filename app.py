import re
from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open("XGB_trained_model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/prediction", methods=["POST"])
def predict():
    D1 = request.form['Age']
    D2 = request.form['Sex']
    D3 = request.form['CPT']
    D4 = request.form['RBP']
    D5 = request.form['SC']
    D6 = request.form['FBS']
    D7 = request.form['RER']
    D8 = request.form['MHR']
    D9 = request.form['EIA']
    D10 = request.form['ST']
    D11 = request.form['SPE']
    D12 = request.form['NMV']
    D13 = request.form['TSR']
    arr = np.array([[D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13]])
    pred = model.predict(arr)
    print(pred[0])
    return render_template("pred.html", data=pred[0])

if __name__=="__main__":
    app.run(debug=True)