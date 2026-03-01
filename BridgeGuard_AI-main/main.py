import os
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple


import torch
import torch.nn as nn
import timm
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from PIL import Image
import cv2

DATASET_ROOT = r"C:\Dataset\CODEBRIM_classification_balanced_dataset\classification_dataset_balanced"
XML_PATH = os.path.join(DATASET_ROOT, "metadata", "defects.xml")  # change if your xml is elsewhere

MODEL_SAVE_PATH = r"image_model.pth"
EPOCHS = 10
BATCH_SIZE = 64
NUM_WORKERS = 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "background",
    "crack",
    "corrosion_stain",
    "efflorescence",
    "exposed_bars",
    "spallation",
]

PRIORITY = [
    "exposed_bars",
    "crack",
    "spallation",
    "corrosion_stain",
    "efflorescence",
    "background",
]

SEVERITY_WEIGHTS = {
    "background": 0.0,
    "crack": 0.9,
    "corrosion_stain": 0.6,
    "efflorescence": 0.2,
    "exposed_bars": 0.85,
    "spallation": 0.7,
}

XML_TAG_TO_CLASS = {
    "Background": "background",
    "Crack": "crack",
    "CorrosionStain": "corrosion_stain",
    "Efflorescence": "efflorescence",
    "ExposedBars": "exposed_bars",
    "Spallation": "spallation",
}

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def list_images_recursive(folder: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f.lower())[1] in exts:
                out.append(os.path.join(root, f))
    return out


def parse_defects_xml(xml_path: str) -> Dict[str, Dict[str, int]]:
    """
    Returns:
      labels_by_filename[basename] = {class_name: 0/1, ...} for all 6 classes.
    The key is the <Defect name="..."> attribute, which is usually the image filename.
    """
    if not os.path.isfile(xml_path):
        raise FileNotFoundError(f"XML not found: {xml_path}")


    tree = ET.parse(xml_path)
    root = tree.getroot()


    labels_by_filename: Dict[str, Dict[str, int]] = {}


    for defect_node in root.findall("Defect"):
        fname = defect_node.attrib.get("name")
        if not fname:
            continue


        lbls = {c: 0 for c in CLASS_NAMES}


        for child in list(defect_node):
            tag = child.tag
            if tag in XML_TAG_TO_CLASS:
                cls = XML_TAG_TO_CLASS[tag]
                try:
                    lbls[cls] = int(str(child.text).strip())
                except Exception:
                    lbls[cls] = 0


        labels_by_filename[fname] = lbls


    return labels_by_filename


def single_label_from_multihot(lbls: Dict[str, int]) -> str:
    """
    Force ONE class per image using PRIORITY.
    If no defect classes are 1, return background.
    """
    any_defect = any(lbls.get(c, 0) == 1 for c in CLASS_NAMES if c != "background")
    if not any_defect:
        return "background"


    for cls in PRIORITY:
        if cls != "background" and lbls.get(cls, 0) == 1:
            return cls


    return "background"


class CodeBrimXmlClassificationDataset(Dataset):
    def __init__(self, split_dir: str, labels_by_filename: Dict[str, Dict[str, int]], transform=None):
        self.transform = transform
        self.class_names = CLASS_NAMES
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}


        all_paths = list_images_recursive(split_dir)
        samples: List[Tuple[str, int]] = []
        skipped_no_label = 0


        for p in all_paths:
            base = os.path.basename(p)
            if base not in labels_by_filename:
                skipped_no_label += 1
                continue


            cls = single_label_from_multihot(labels_by_filename[base])
            samples.append((p, self.class_to_idx[cls]))


        if len(samples) == 0:
            raise RuntimeError(
                f"No labeled images found in split folder: {split_dir}\n"
                f"Common causes:\n"
                f" - XML <Defect name='...'> values don't match image basenames\n"
                f" - XML_PATH points to the wrong defects.xml\n"
                f" - your images are not actually inside this split folder\n"
            )


        self.samples = samples
        if skipped_no_label > 0:
            print(f"[{os.path.basename(split_dir)}] Skipped {skipped_no_label} images not found in XML labels.")


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y


def build_image_model(num_classes: int):
    model = timm.create_model(
        "efficientnet_b3",
        pretrained=True,
        num_classes=0,
        global_pool="avg"
    )

    for name, param in model.named_parameters():
        if any(x in name for x in ["blocks.0", "blocks.1", "blocks.2"]):
            param.requires_grad = False


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


def make_loaders(dataset_root: str, xml_path: str, batch_size: int, num_workers: int):
    train_dir = os.path.join(dataset_root, "train")
    val_dir   = os.path.join(dataset_root, "val")
    test_dir  = os.path.join(dataset_root, "test")


    for p in (train_dir, val_dir, test_dir):
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Missing folder: {p}")


    labels_by_filename = parse_defects_xml(xml_path)


    train_ds = CodeBrimXmlClassificationDataset(train_dir, labels_by_filename, transform=train_transforms)
    val_ds   = CodeBrimXmlClassificationDataset(val_dir, labels_by_filename, transform=val_transforms)
    test_ds  = CodeBrimXmlClassificationDataset(test_dir, labels_by_filename, transform=val_transforms)


    missing = [c for c in CLASS_NAMES if c not in SEVERITY_WEIGHTS]
    if missing:
        raise ValueError(f"SEVERITY_WEIGHTS missing keys for classes: {missing}")


    pin = torch.cuda.is_available()


    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)


    print("Class mapping (fixed):", {c: i for i, c in enumerate(CLASS_NAMES)})
    print(f"Train/Val/Test sizes: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")


    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader, CLASS_NAMES


def train_image_model(dataset_root: str, xml_path: str, epochs: int, save_path: str,
                      batch_size: int, num_workers: int):
    train_ds, val_ds, _, train_loader, val_loader, _, class_names = make_loaders(
        dataset_root, xml_path, batch_size, num_workers
    )


    model = build_image_model(num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


    best_val_acc = 0.0


    patience = 4
    epochs_no_improve = 0


    with open(save_path + ".classes.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)


    for epoch in range(epochs):
        model.train()
        train_correct = 0


        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_correct += (outputs.argmax(1) == labels).sum().item()


        model.eval()
        val_correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_correct += (outputs.argmax(1) == labels).sum().item()


        train_acc = train_correct / len(train_ds)
        val_acc = val_correct / len(val_ds)
        scheduler.step()


        print(f"Epoch {epoch+1:02d}/{epochs} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0  


            torch.save(model.state_dict(), save_path)
            print(f"Saved best model (val_acc={val_acc:.3f}) -> {save_path}")


        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{patience} epochs")


            if epochs_no_improve >= patience:
                print("\n Early stopping triggered")
                break


    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    return model, class_names


def load_image_model(save_path: str):
    classes_path = save_path + ".classes.json"
    if not os.path.isfile(classes_path):
        raise FileNotFoundError(f"Missing {classes_path}. Train first.")


    with open(classes_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)


    model = build_image_model(num_classes=len(class_names))
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()
    return model, class_names


def evaluate_on_test(dataset_root: str, xml_path: str, save_path: str, batch_size: int, num_workers: int):
    _, _, test_ds, _, _, test_loader, class_names = make_loaders(
        dataset_root, xml_path, batch_size, num_workers
    )
    model, _ = load_image_model(save_path)
    model = model.to(device)


    correct = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()


    acc = correct / len(test_ds)
    print(f"Test accuracy: {acc:.3f}")
    return acc, class_names


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
        sum(all_probs[c] * SEVERITY_WEIGHTS[c] for c in class_names) * 100.0,
        100.0
    )


    return {
        "predicted_class": pred_class,
        "confidence": confidence,
        "all_probs": all_probs,
        "heatmap": heatmap,
        "visual_risk_score": round(visual_risk_score, 2),
    }


if __name__ == "__main__":
    model, class_names = train_image_model(
        DATASET_ROOT,
        xml_path=XML_PATH,
        epochs=EPOCHS,
        save_path=MODEL_SAVE_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )


    evaluate_on_test(
        DATASET_ROOT,
        xml_path=XML_PATH,
        save_path=MODEL_SAVE_PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )


    TEST_IMAGE = r"C:\Dataset\CODEBRIM_classification_balanced_dataset\classification_dataset_balanced\test\defects\image_0000675_crop_0000005.png"


    model, class_names = load_image_model(MODEL_SAVE_PATH)
    model = model.to(device)


    result = run_image_inference(TEST_IMAGE, model, class_names)


    print("\n=== Inference Result ===")
    print("Predicted class:", result["predicted_class"])
    print("Confidence:", result["confidence"])
    print("Risk score:", result["visual_risk_score"])
    print("Top-3 probs:", sorted(result["all_probs"].items(), key=lambda x: x[1], reverse=True)[:3])


    heatmap_rgb = result["heatmap"]
    heatmap_bgr = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite("heatmap_output.jpg", heatmap_bgr)
    print("Saved heatmap to heatmap_output.jpg")

