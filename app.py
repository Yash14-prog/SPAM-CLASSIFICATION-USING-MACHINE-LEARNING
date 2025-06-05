from flask import Flask, request, jsonify, render_template
import json
import time
import random
import pickle
from flask_cors import CORS
from threading import Thread, Lock

# Load Flask app
app = Flask(__name__)
CORS(app)

# Load the emails from a JSON file
with open("emails.json") as f:
    __e = json.load(f)

__c = []  # Store processed emails and predictions
__i = 0   # Index for traversing emails
__l = Lock()  # Thread lock

# Load the trained model and vectorizer
with open("MNB.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Function to classify an email as spam or ham using the ML model
def __cls(text):
    vect_text = vectorizer.transform([text])
    pred = model.predict(vect_text)
    return "s" if pred[0] == 1 else "h"  # Assuming spam is labeled as 1

# Background thread to simulate real-time email processing
def __th():
    global __i
    while __i < len(__e):
        with __l:
            e = __e[__i]
            __i += 1
        t = e.get("text", "")
        lbl = e.get("label", None)
        p = __cls(t) if lbl is None else ("s" if lbl == "spam" else "h")
        with __l:
            __c.append({"text": t, "prediction": "spam" if p == "s" else "ham"})
        time.sleep(20)  # Simulate delay between messages

# Endpoint to fetch the latest prediction
@app.route("/random_email")
def rnd():
    with __l:
        if __c:
            return jsonify(__c[-1])
        return jsonify({"text": "Waiting for next message...", "prediction": "ham"})

# Endpoint to manually classify an email
@app.route("/predict", methods=["POST"])
def pred():
    d = request.json
    t = d.get("email", "")
    p = __cls(t)
    res = {"text": t, "prediction": "spam" if p == "s" else "ham"}
    with __l:
        __c.append(res)
    return jsonify(res)

# Render the homepage
@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    Thread(target=__th, daemon=True).start()  # Start background email classifier
    app.run(debug=True, port=5000)

