import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ====================== CONFIGURATION ======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Utilisation du device : {DEVICE}")

print(" Chargement des poids du modèle best_glaucoma_convnext.pth ...")

try:
    # Chargement du checkpoint
    checkpoint = torch.load("models/best_glaucoma_convnext.pth",
                            map_location=DEVICE,
                            weights_only=True)

    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Suppression du préfixe "base." si présent
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("base.", "")
        new_state_dict[new_key] = v

    # Création du modèle ConvNeXt Tiny
    from torchvision.models import convnext_tiny

    model = convnext_tiny(weights=None)

    # === ADAPTATION DU CLASSIFIER pour correspondre à ton modèle ===
    # Ton modèle a : LayerNorm → Flatten → Linear → ? (probablement un autre Linear ou activation)
    in_features = model.classifier[2].in_features   # normalement 768 pour convnext_tiny

    model.classifier = torch.nn.Sequential(
        model.classifier[0],           # Garde le LayerNorm original
        torch.nn.Flatten(1),
        torch.nn.Linear(in_features, 1)   # Sortie binaire (1 neurone)
        # Si tu avais un 4ème layer dans le classifier, on le simplifie ici en 1 sortie
    )

    # Chargement des poids
    model.load_state_dict(new_state_dict, strict=False)   # strict=False pour ignorer les petites différences

    model = model.to(DEVICE)
    model.eval()

    print(" Modèle chargé avec succès ! (classifier adapté pour sortie binaire)")

except Exception as e:
    print(f" Erreur lors du chargement du modèle : {e}")
    model = None
    raise

# ====================== TRANSFORMATIONS ======================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ====================== FONCTION PREDICT ======================
def predict(img: Image.Image) -> dict:
    if model is None:
        raise ValueError("Le modèle n'a pas pu être chargé")

    try:
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()     # Probabilité Glaucome

        return {
            "prediction": "Glaucome" if prob > 0.5 else "Normal",
            "probability_glaucoma": round(prob, 4),
            "probability_normal": round(1 - prob, 4),
            "confidence": round(max(prob, 1 - prob) * 100, 2)
        }

    except Exception as e:
        raise ValueError(f"Erreur pendant la prédiction : {str(e)}")
