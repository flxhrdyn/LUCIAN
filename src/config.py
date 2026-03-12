from pathlib import Path

# Project root (one level above this file)
BASE_DIR = Path(__file__).resolve().parent.parent

# Model configuration
MODEL_URL = "https://huggingface.co/felixhrdyn/convnextv1-lung-cancer/resolve/main/convnext_lung_82.keras"
MODEL_PATH = str(BASE_DIR / "convnext_lung_82.keras")
IMAGE_SIZE = (224, 224)

# Classification labels
CLASS_LABELS_EN = ["Adenocarcinoma", "Benign Tissue", "Squamous Cell Carcinoma"]
CLASS_LABELS_ID = ["Adenokarsinoma Paru", "Jaringan Paru Jinak", "Karsinoma Sel Skuamosa Paru"]
CLASS_COLORS = ["#e74c3c", "#2ecc71", "#e67e22"]  # Red, Green, Orange

# Grad-CAM configuration
# `flatten` sits immediately after `convnext_base` in the outer model.
# Its `.input` tensor is the 7×7×1024 feature map used for Grad-CAM.
GRADCAM_LAYER = "flatten"

# Asset paths
ASSETS_82_SPLIT = str(BASE_DIR / "assets" / "model_performance_82split")
ASSETS_73_SPLIT = str(BASE_DIR / "assets" / "model_performance_73split")

# Demo sample images (one representative per class)
_DEMO_DIR = BASE_DIR / "assets" / "lung_cancer_image_demo"
DEMO_IMAGES = {
    "LUAD Sample":   str(_DEMO_DIR / "adenocarcinoma"        / "adenocarcinoma1.jpg"),
    "Benign Sample": str(_DEMO_DIR / "benign"                / "benign1.jpg"),
    "LUSC Sample":   str(_DEMO_DIR / "squamous_cell_carcinoma"/ "squamous1.jpg"),
}
