# backend/app_diag1.py
from flask import Flask

app = Flask(__name__)

@app.route("/")
def index():
    return "<h1>It works 🎉</h1><p>This is inline HTML</p>"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
