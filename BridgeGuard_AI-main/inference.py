import os
import json
import argparse

import torch
import torch.nn as nn
import timm
import numpy as np
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import cv2


MODEL_SAVE_PATH = "image_model.pth" 

SEVERITY_WEIGHTS = {
    "background": 0.0,
    "crack": 0.9,
    "corrosion_stain": 0.6,
    "efflorescence": 0.2,
    "exposed_bars": 0.85,
    "spallation": 0.7,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def build_image_model(num_classes: int):
    model = timm.create_model(
        "efficientnet_b3",
        pretrained=False,   
        num_classes=0,
        global_pool="avg"
    )

    in_features = model.num_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )

    return model.to(device)


def load_image_model(save_path: str):
    classes_path = save_path + ".classes.json"
    if not os.path.isfile(save_path):
        raise FileNotFoundError(f"Missing model weights: {save_path}")
    if not os.path.isfile(classes_path):
        raise FileNotFoundError(f"Missing classes file: {classes_path} (train first)")

    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    model = build_image_model(num_classes=len(class_names))
    state = torch.load(save_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, class_names


def run_image_inference(image_path: str, model, class_names: list[str]):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (224, 224))

    pil_image = Image.fromarray(image_resized)
    tensor = val_transforms(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])
    all_probs = {class_names[i]: float(probs[i]) for i in range(len(class_names))}

    target_layers = [model.blocks[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(pred_idx)]
    grayscale_cam = cam(input_tensor=tensor, targets=targets)

    image_float = image_resized / 255.0
    heatmap = show_cam_on_image(image_float, grayscale_cam[0], use_rgb=True)  
    visual_risk_score = min(
        sum(all_probs[c] * SEVERITY_WEIGHTS.get(c, 0.0) for c in class_names) * 100.0,
        100.0
    )

    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "all_probs": all_probs,
        "heatmap": heatmap, 
        "visual_risk_score": round(visual_risk_score, 2),
    }


def save_outputs(image_path: str, result: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(image_path))[0]
    heatmap_path = os.path.join(out_dir, f"{base}_heatmap.jpg")
    json_path = os.path.join(out_dir, f"{base}_result.json")

    heatmap_rgb = result["heatmap"]
    heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(heatmap_path, heatmap_bgr)

    result_for_json = dict(result)
    result_for_json.pop("heatmap", None)
    result_for_json["image_path"] = image_path

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_for_json, f, indent=2)

    return heatmap_path, json_path


def is_image_file(p: str) -> bool:
    ext = os.path.splitext(p.lower())[1]
    return ext in [".jpg", ".jpeg", ".png", ".bmp"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_SAVE_PATH, help="Path to image_model.pth")
    parser.add_argument("--image", default=None, help="Path to one image")
    parser.add_argument("--folder", default=None, help="Path to a folder of images (optional)")
    parser.add_argument("--out", default="inference_outputs", help="Output folder for heatmaps/results")
    args = parser.parse_args()

    if args.image is None and args.folder is None:
        raise SystemExit("Provide --image PATH or --folder PATH")

    model, class_names = load_image_model(args.model)
    model = model.to(device)

    print("Device:", device)
    print("Classes:", class_names)

    paths = []
    if args.image:
        paths.append(args.image)
    if args.folder:
        for root, _, files in os.walk(args.folder):
            for f in files:
                p = os.path.join(root, f)
                if is_image_file(p):
                    paths.append(p)

    for p in paths:
        result = run_image_inference(p, model, class_names)

        print("\n=== Result ===")
        print("Image:", p)
        print("Predicted:", result["predicted_class"])
        print("Confidence:", result["confidence"])
        print("Risk score:", result["visual_risk_score"])
        top3 = sorted(result["all_probs"].items(), key=lambda x: x[1], reverse=True)[:3]
        print("Top-3:", top3)

        heatmap_path, json_path = save_outputs(p, result, args.out)
        print("Saved heatmap:", heatmap_path)
        print("Saved json:", json_path)


if __name__ == "__main__":
    main()