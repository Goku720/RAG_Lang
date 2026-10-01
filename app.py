import os

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

import rag

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)


@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/documents", methods=["GET"])
def documents():
    return jsonify({"documents": rag.list_documents()})


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("pdf")
    if not file or not file.filename:
        return jsonify({"error": "No file provided."}), 400
    name = secure_filename(file.filename)
    if not name.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported."}), 400
    path = os.path.join(UPLOAD_DIR, name)
    file.save(path)
    try:
        n = rag.ingest_pdf(path, name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception as e:
        app.logger.exception("Ingest failed")
        return jsonify({"error": f"Ingest failed: {e}"}), 500
    return jsonify({"message": "Ingested.", "source": name, "chunks": n})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("query") or "").strip()
    source = data.get("source")
    if not question or not source:
        return jsonify({"error": "Both 'query' and 'source' are required."}), 400
    try:
        return jsonify(rag.ask(question, source))
    except Exception as e:
        app.logger.exception("Ask failed")
        return jsonify({"error": f"Could not answer: {e}"}), 502


if __name__ == "__main__":
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")),
            debug=os.getenv("FLASK_DEBUG") == "1")
