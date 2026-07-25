import numpy as np
from flask import Flask, request, render_template
import sqlite3
import random
import smtplib
from email.message import EmailMessage

# ML / Image imports (kept for future use)
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend as K

# ======================
# Custom metric functions (kept for future model use)
# ======================

def specificity_m(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def sensitivity_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def mae(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

def mse(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

# ======================
# Flask App
# ======================

app = Flask(__name__)

# ======================
# Signup with OTP
# ======================

@app.route("/signup")
def signup():
    global otp, username, name, email, number, password

    username = request.args.get('t1', '')
    name = request.args.get('t2', '')
    email = request.args.get('t3', '')
    number = request.args.get('t4', '')
    password = request.args.get('t5', '')

    otp = random.randint(1000, 5000)
    print("OTP:", otp)

    try:
        msg = EmailMessage()
        msg.set_content("Your OTP is : " + str(otp))
        msg['Subject'] = 'OTP'
        msg['From'] = "vandhanatruprojects@gmail.com"
        msg['To'] = email

        s = smtplib.SMTP('smtp.gmail.com', 587)
        s.starttls()
        s.login("vandhanatruprojects@gmail.com", "pahksvxachlnoopc")
        s.send_message(msg)
        s.quit()
    except Exception as e:
        print("Email error:", e)

    return render_template("val.html")

# ======================
# OTP Verification
# ======================

@app.route('/predict_lo', methods=['POST'])
def predict_lo():
    global otp, username, name, email, number, password

    message = request.form['t1']

    if int(message) == otp:
        con = sqlite3.connect('signup.db')
        cur = con.cursor()
        cur.execute(
            "INSERT INTO info (user, email, password, mobile, name) VALUES (?, ?, ?, ?, ?)",
            (username, email, password, number, name)
        )
        con.commit()
        con.close()
        return render_template("signin.html")

    return render_template("signup.html")

# ======================
# Signin
# ======================

@app.route("/signin")
def signin():
    global username1

    mail1 = request.args.get('t1', '')
    password1 = request.args.get('t2', '')

    con = sqlite3.connect('signup.db')
    cur = con.cursor()
    cur.execute(
        "SELECT user, password FROM info WHERE user=? AND password=?",
        (mail1, password1)
    )
    data = cur.fetchone()
    con.close()

    username1 = mail1

    if data is None:
        return render_template("signin.html")

    return render_template("home.html")

# ======================
# Prediction (DUMMY MODEL)
# ======================

@app.route('/predict', methods=['POST'])
def predict():
    if 'files' not in request.files:
        return "No file uploaded"

    image_file = request.files['files']
    if image_file.filename == '':
        return "No file selected"

    image_path = 'temp_image.jpg'
    image_file.save(image_path)

    # Image preprocessing (for future model)
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # 🔴 DUMMY PREDICTION (NO MODEL)
    result = random.randint(0, 25)

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
        25: 'Oral Squamous Cell Carcinoma'
    }

    output = result_mapping.get(result, "Unknown")
    return render_template('prediction.html', output=output)

# ======================
# Page Routes
# ======================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/Logon')
def logon():
    return render_template('signup.html')

@app.route('/Login')
def login():
    return render_template('signin.html')

# ======================
# Run App
# ======================

if __name__ == "__main__":
    app.run(debug=True)
