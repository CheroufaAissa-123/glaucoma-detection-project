import io
import base64
import logging
from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
from flask_cors import CORS

# ====================== IMPORT DU MODÈLE ======================
from load_model import predict  # Assure-toi que ton predict est bien défini

# ====================== CONFIGURATION ======================
app = Flask(__name__)
CORS(app)  # Autorise les requêtes depuis le navigateur (utile pour démo)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB max

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "tiff"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ====================== UTILS ======================
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def load_image(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(file_bytes))
        img.verify()
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        return img
    except UnidentifiedImageError:
        raise ValueError("Image invalide ou corrompue")
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement de l'image: {str(e)}")


# ====================== ROUTES ======================
@app.route("/health", methods=["GET"])
def health():
 import torch
 from datetime import datetime

 return jsonify({
  "success": True,
  "status": "healthy",
  "service_name": "Glaucoma Detection API",
  "version": "1.0-demo",
  "description": "Détection automatique de glaucome à partir de photos du fond d'œil",

  "model": {
   "filename": "best_glaucoma_convnext.pth",
   "type": "ConvNeXt",
   "task": "Classification binaire",
   "classes": ["Normal", "Glaucome"],
   "input_shape": [3, 224, 224],
   "device": "CUDA" if torch.cuda.is_available() else "CPU"
  },

  "api": {
   "framework": "Flask 3.x",
   "cors_enabled": True,
   "max_image_size_mb": 10,
   "supported_formats": ["png", "jpg", "jpeg", "bmp", "tiff"]
  },

  "endpoints": {
   "health_check": "/health",
   "predict_from_file": "/predict",
   "predict_from_base64": "/predict/base64"
  },

  "server_info": {
   "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
   "hostname": __import__('socket').gethostname(),
   "environment": "local_demo"
  },

  "message": "L'API fonctionne correctement. Vous pouvez maintenant tester les prédictions.",
  "documentation_tip": "Utilisez Postman ou le fichier HTML de test pour envoyer des images."
 }), 200


@app.route("/predict", methods=["POST"])
def predict_file():
    if "file" not in request.files:
        return jsonify({"error": "Clé 'file' manquante"}), 400

    file = request.files["file"]
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Format non supporté (png, jpg, jpeg, bmp, tiff seulement)"}), 400

    try:
        img = load_image(file.read())
        result = predict(img)  # Appel à ton modèle
        return jsonify(result), 200
    except ValueError as e:
        logger.warning(str(e))
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Erreur inattendue: {str(e)}", exc_info=True)
        return jsonify({"error": "Erreur interne du serveur"}), 500


@app.route("/predict/base64", methods=["POST"])
def predict_b64():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "JSON avec la clé 'image' (base64) requis"}), 400

    try:
        # Support data:image/...;base64, ou juste le base64
        image_data = data["image"].split(",")[-1]
        img_bytes = base64.b64decode(image_data)
        img = load_image(img_bytes)
        result = predict(img)
        return jsonify(result), 200
    except ValueError as e:
        logger.warning(str(e))
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(str(e), exc_info=True)
        return jsonify({"error": "Erreur interne du serveur"}), 500


# ====================== LANCEMENT ======================
if __name__ == "__main__":
    print(" Démarrage de l'API Glaucome en mode démo locale...")
    print("   → Accède à http://127.0.0.1:5000/health pour tester")
    print("   → Utilise /predict ou /predict/base64 pour les prédictions")
    app.run(host="127.0.0.1", port=5000, debug=True)  # debug=True est OK pour démo locale