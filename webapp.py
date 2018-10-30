import os
from flask import Flask, request, jsonify, abort
from model_runtime.model_wrappers import initialize_model

app = Flask(__name__)
model = initialize_model(os.environ.get("MODEL_DIR"))


@app.route("/classify", methods=['POST'])
def classify_request_handler():
    request_payload = request.get_json(force=True, silent=True)
    if not request_payload or "text" not in request_payload.keys():
        abort(400)
    else:
        text = str(request_payload["text"])
        return jsonify({"class": model.predict_sentiment([text])[0]})


if __name__ == "__main__":
    port = int(os.environ.get("WEB_APP_PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
