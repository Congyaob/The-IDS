# backend/app_probe.py
from flask import Flask, request, jsonify
import sys

app = Flask(__name__)

@app.before_request
def _log_request():
    print(f" Incoming: {request.method} {request.path}", file=sys.stderr, flush=True)

@app.route("/")
def root():
    return "It works (from app_probe.py /)", 200

@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    print(" Expecting these routes:")
    print(app.url_map)  
    app.run(debug=False, use_reloader=False, host="127.0.0.1", port=5055)
