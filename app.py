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
    D1 = request.form['Cc']
    D2 = request.form['CDI']
    D3 = request.form['Gen']
    D4 = request.form['RE']
    D5 = request.form['EU']
    D6 = request.form['EL']
    D7 = request.form['MD']
    D8 = request.form['Ex']
    D9 = request.form['CS']
    D10 = request.form['CT']
    D11 = request.form['LNJ']
    D12 = request.form['TH']
    arr = np.array([[D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12]])
    pred = model.predict(arr)
    return render_template("pred.html", data=pred[0])

if __name__=="__main__":
    app.run(debug=True)