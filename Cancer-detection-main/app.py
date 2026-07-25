import numpy as np
from flask import Flask, request, render_template
import sqlite3
import random
import smtplib
from email.message import EmailMessage

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend as K

# =========================
# CUSTOM METRICS (SAFE)
# =========================
def specificity_m(y_true, y_pred):
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    pn = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return tn / (pn + K.epsilon())

def sensitivity_m(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))
    return tp / (pp + K.epsilon())

# =========================
# LOAD MODEL
# =========================
MODEL_PATH = "models/xception.h5"

model = load_model(
    MODEL_PATH,
    custom_objects={
        "specificity_m": specificity_m,
        "sensitivity_m": sensitivity_m
    }
)

print("✅ Model loaded successfully")

# =========================
# CLASS MAPPING (8 CLASSES)
# MUST MATCH TRAINING ORDER
# =========================
CLASS_NAMES = {
    0: "ALL",
    1: "Brain Cancer",
    2: "Breast Cancer",
    3: "Cervical Cancer",
    4: "Kidney Cancer",
    5: "Lung and Colon Cancer",
    6: "Lymphoma",
    7: "Oral Cancer"
}

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

# -------------------------
# SIGNUP
# -------------------------
@app.route("/signup")
def signup():
    global otp, username, name, email, number, password

    username = request.args.get("t1", "")
    name = request.args.get("t2", "")
    email = request.args.get("t3", "")
    number = request.args.get("t4", "")
    password = request.args.get("t5", "")

    otp = random.randint(1000, 5000)

    msg = EmailMessage()
    msg.set_content("Your OTP is : " + str(otp))
    msg["Subject"] = "OTP"
    msg["From"] = "vandhanatruprojects@gmail.com"
    msg["To"] = email

    s = smtplib.SMTP("smtp.gmail.com", 587)
    s.starttls()
    s.login("vandhanatruprojects@gmail.com", "pahksvxachlnoopc")
    s.send_message(msg)
    s.quit()

    return render_template("val.html")

# -------------------------
# OTP VERIFY
# -------------------------
@app.route("/predict_lo", methods=["POST"])
def predict_lo():
    global otp, username, name, email, number, password

    message = request.form["t1"]
    if int(message) == otp:
        con = sqlite3.connect("signup.db")
        cur = con.cursor()
        cur.execute(
            "INSERT INTO info (user, email, password, mobile, name) VALUES (?, ?, ?, ?, ?)",
            (username, email, password, number, name),
        )
        con.commit()
        con.close()
        return render_template("signin.html")

    return render_template("signup.html")

# -------------------------
# SIGNIN
# -------------------------
@app.route("/signin")
def signin():
    mail1 = request.args.get("t1", "")
    password1 = request.args.get("t2", "")

    con = sqlite3.connect("signup.db")
    cur = con.cursor()
    cur.execute(
        "SELECT user, password FROM info WHERE user=? AND password=?",
        (mail1, password1),
    )
    data = cur.fetchone()
    con.close()

    if data:
        return render_template("home.html")
    return render_template("signin.html")

# -------------------------
# PREDICTION (REAL MODEL)
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "files" not in request.files:
        return "No file uploaded"

    image_file = request.files["files"]
    if image_file.filename == "":
        return "No file selected"

    image_path = "temp_image.jpg"
    image_file.save(image_path)

    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    class_index = np.argmax(preds)
    confidence = np.max(preds)

    result = CLASS_NAMES.get(class_index, "Unknown")

    return render_template(
        "prediction.html",
        output=result,
        confidence=f"{confidence*100:.2f}%"
    )

# -------------------------
# ROUTES
# -------------------------
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

# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
