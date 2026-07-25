import numpy as np
from flask import Flask, request, render_template
import sqlite3
import random
import smtplib
from email.message import EmailMessage

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow.keras.backend as K

from sklearn.metrics import f1_score, recall_score, precision_score

# -------------------- CUSTOM METRICS --------------------

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

# -------------------- LOAD MODEL --------------------

model_path = "models/xception.h5"

custom_objects = {
    "f1_score": f1_score,
    "recall_m": recall_score,
    "precision_m": precision_score,
    "specificity_m": specificity_m,
    "sensitivity_m": sensitivity_m,
    "mae": mae,    ##Mean Absolute Error
    "mse": mse.    ##Mean Squared Error
}

model = load_model(model_path, custom_objects=custom_objects)

# -------------------- FLASK APP --------------------

app = Flask(__name__)

# -------------------- SIGNUP (NO OTP PAGE) --------------------

@app.route("/signup")
def signup():
    username = request.args.get("t1", "")
    name = request.args.get("t2", "")
    email = request.args.get("t3", "")
    number = request.args.get("t4", "")
    password = request.args.get("t5", "")

    con = sqlite3.connect("signup.db")
    cur = con.cursor()
    cur.execute(
        "INSERT INTO info (user, email, password, mobile, name) VALUES (?, ?, ?, ?, ?)",
        (username, email, password, number, name),
    )
    con.commit()
    con.close()

    return render_template("signin.html")

# -------------------- SIGNIN --------------------

@app.route("/signin")
def signin():
    mail = request.args.get("t1", "")
    password = request.args.get("t2", "")

    con = sqlite3.connect("signup.db")
    cur = con.cursor()
    cur.execute(
        "SELECT user, password FROM info WHERE user=? AND password=?",
        (mail, password),
    )
    data = cur.fetchone()
    con.close()

    if data:
        return render_template("home.html")
    else:
        return render_template("signin.html")

# -------------------- PREDICTION --------------------

@app.route("/predict", methods=["POST"])
def predict():
    if "files" not in request.files:
        return "No file uploaded"

    image_file = request.files["files"]
    image_file.save("temp.jpg")

    img = load_img("temp.jpg", target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    result = np.argmax(model.predict(img))

    classes = [
        "All Benign", "All Early", "All Pre", "All Pro",
        "Brain Glioma", "Brain Meningioma", "Brain Tumor",
        "Breast Benign", "Breast Malignant",
        "Cervix Dyskeratotic", "Cervix Koilocytotic",
        "Cervix Metaplastic", "Cervix Parabasal",
        "Cervix Superficial Intermediate",
        "Colon Adenocarcinoma", "Colon Benign",
        "Kidney Normal", "Kidney Tumor",
        "Lung Adenocarcinoma", "Lung Benign",
        "Lung Squamous Cell Carcinoma",
        "Lymph CLL", "Lymph Follicular",
        "Lymph Mantle", "Oral Normal",
        "Oral Squamous Cell Carcinoma"
    ]

    return render_template("prediction.html", output=classes[result])

# -------------------- ROUTES --------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/Login")
def login():
    return render_template("signin.html")

@app.route("/Logon")
def logon():
    return render_template("signup.html")

# -------------------- RUN --------------------

if __name__ == "__main__":
    app.run(debug=True)
