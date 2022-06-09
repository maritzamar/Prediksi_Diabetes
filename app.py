
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pickle
import requests
import os


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

dataset = pd.read_csv('diabetes.csv')

dataset_X = dataset.iloc[:,[1, 2, 5, 7]].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
dataset_scaled = sc.fit_transform(dataset_X)

picFolder = os.path.join('static')
print(picFolder)
app.config['UPLOAD_FOLDER'] = picFolder

nama1=[]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sub', methods = ['GET','POST'])

def submit():
    name = request.form['nama']
    nama1.append(name)
    text = 'SILAHKAN MENGISI FORM UNTUK MELAKUKAN PREDIKSI DIABETES'
    return render_template('check1.html', prediction_text =text, nama=nama1)


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict( sc.transform(final_features) )

    pic1 = os.path.join(app.config['UPLOAD_FOLDER'], 'Untitled-1.png')
    
    if prediction == 1:
        pred = "Wah Kondisi anda kurang baik, segera pergi ke pusat kesehatan untuk prosedur pemeriksaan lebih lanjut. Anda Memungkinkan Positif Diabetes, "
        # return redirect(url_for(pred, name = nama))
    elif prediction == 0:
        pred = " Wah Kondisi anda baik, jangan lupa jaga kesehatan ya."
    output = pred

    return render_template('check1.html', prediction_text= '{}'.format(output), nama=nama1, userimage=pic1)
    # return render_template(submit, prediction_text= '{}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)