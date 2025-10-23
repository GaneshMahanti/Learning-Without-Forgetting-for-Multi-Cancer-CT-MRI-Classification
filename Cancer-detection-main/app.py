import numpy as np
from flask import Flask, request, jsonify, render_template
import sqlite3
import pandas as pd
import pickle
import random
import smtplib 
from email.message import EmailMessage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def specificity_m(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    specificity = true_negatives / (possible_negatives + K.epsilon())
    return specificity

def sensitivity_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    sensitivity = true_positives / (possible_positives + K.epsilon())
    return sensitivity

def mae(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

def mse(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

model_path2 = 'models/xception.h5' 

custom_objects = {
    'f1_score': f1_score,
    'recall_m': recall_score,
    'precision_m': precision_score,
    'specificity_m': specificity_m,
    'sensitivity_m': sensitivity_m,
    'mae' : mae,
    'mse' : mse
}


model = load_model(model_path2, custom_objects=custom_objects)


app = Flask(__name__)

@app.route("/signup")
def signup():
    global otp, username, name, email, number, password
    username = request.args.get('t1','')
    name = request.args.get('t2','')
    email = request.args.get('t3','')
    number = request.args.get('t4','')
    password = request.args.get('t5','')
    otp = random.randint(1000,5000)
    print(otp)
    msg = EmailMessage()
    msg.set_content("Your OTP is : "+str(otp))
    msg['Subject'] = 'OTP'
    msg['From'] = "vandhanatruprojects@gmail.com"
    msg['To'] = email
    
    
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("vandhanatruprojects@gmail.com", "pahksvxachlnoopc")
    s.send_message(msg)
    s.quit()
    return render_template("val.html")

@app.route('/predict_lo', methods=['POST'])
def predict_lo():
    global otp, username, name, email, number, password
    if request.method == 'POST':
        message = request.form['t1']
        print(message)
        if int(message) == otp:
            print("TRUE")
            con = sqlite3.connect('signup.db')
            cur = con.cursor()
            cur.execute("insert into `info` (`user`,`email`, `password`,`mobile`,`name`) VALUES (?, ?, ?, ?, ?)",(username,email,password,number,name))
            con.commit()
            con.close()
            return render_template("signin.html")
    return render_template("signup.html")

@app.route("/signin")
def signin():
    global username1
    mail1 = request.args.get('t1','')
    password1 = request.args.get('t2','')
    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute("select `user`, `password` from info where `user` = ? AND `password` = ?",(mail1,password1,))
    data = cur.fetchone()
    username1 = mail1
    if data == None:
        return render_template("signin.html")
    elif mail1 == 'admin' and password1 == 'admin':
        return render_template("home.html")    
    elif mail1 == str(data[0]) and password1 == str(data[1]):
        return render_template("home.html")
    else:
        return render_template("signin.html")




@app.route('/predict', methods=['POST'])
def predict():
    if 'files' in request.files:
        image_file = request.files['files']
        if image_file.filename != '':
            
            image_path = 'temp_image.jpg'
            image_file.save(image_path)

            
            image = load_img(image_path, target_size=(224, 224))
            image = img_to_array(image)
            image = image / 255
            image = np.expand_dims(image, axis=0)

            
            result = np.argmax(model.predict(image))

            result_mapping = {
            0: 'All Benign',
            1: 'All Early',
            2: 'All Pre',
            3: 'All Pro',
            4: 'Brain Glioma',
            5: 'Brain Meningioma',
            6: 'Brain Tumor',
            7: 'Breast Benign',
            8: 'Breast Malignant',
            9: 'Cervix Dyskeratotic',
            10: 'Cervix Koilocytotic',
            11: 'Cervix Metaplastic',
            12: 'Cervix Parabasal',
            13: 'Cervix Superficial Intermediate',
            14: 'Colon Adenocarcinoma',
            15: 'Colon Benign',
            16: 'Kidney Normal',
            17: 'Kidney Tumor',
            18: 'Lung Adenocarcinoma',
            19: 'Lung Benign',
            20: 'Lung Squamous Cell Carcinoma',
            21: 'Lymph Chronic Lymphocytic Leukemia',
            22: 'Lymph Follicular Lymphoma',
            23: 'Lymph Mantle Cell Lymphoma',
            24: 'Oral Normal',
            25: 'Oral Squamous Cell Carcinoma'}

            if result in result_mapping:
                result_all = result_mapping[result]
            else:
                result_all = "Unknown"

            return render_template('prediction.html', output=result_all)
    return "No file uploaded."

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/notebook")
def notebook():
    return render_template("Notebook.html")

@app.route("/notebook1")
def notebook1():
    return render_template("Notebook1.html")

@app.route("/notebook2")
def notebook2():
    return render_template("Notebook2.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route('/home')
def home():
	return render_template('home.html')


@app.route('/Logon')
def logon():
	return render_template('signup.html')

@app.route('/Login')
def login():
	return render_template('signin.html')

if __name__ == "__main__":
    app.run(debug=True)
