
# main_patchcore.py
import os
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from types import SimpleNamespace
from anomalib.data.datasets.image.mvtecad import MVTecADDataset
import anomalib.data.datasets.image.mvtecad as mvtec_module
from anomalib.models import Patchcore
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader
from torchvision import transforms
from anomalib.models.image.patchcore.lightning_model import Patchcore
from anomalib.data.dataclasses.torch import InferenceBatch
from anomalib.post_processing.post_processor import PostProcessor

# --- Patches for Safe Inference ---
def safe_normalize_batch(self, batch):
    anomaly_map = batch.anomaly_map
    anomaly_map = self._normalize(anomaly_map, self.pixel_min, self.pixel_max, self.pixel_threshold)
    return InferenceBatch(batch.pred_score, batch.pred_label, batch.pred_mask, anomaly_map)
PostProcessor.normalize_batch = safe_normalize_batch

def safe_threshold_batch(self, batch):
    pred_score = batch.pred_score
    pred_label = pred_score > self.image_threshold if pred_score is not None else None
    return InferenceBatch(pred_score, pred_label, batch.pred_mask, batch.anomaly_map)
PostProcessor.threshold_batch = safe_threshold_batch

def patch_predict_step(self, batch, batch_idx):
    outputs = self.model(batch["image"])
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    if isinstance(outputs, InferenceBatch):
        if outputs.pred_score is None:
            outputs.pred_score = torch.zeros(len(batch["image"]))
        return outputs
    if isinstance(outputs, dict):
        pred_score = outputs.get("pred_score") or outputs.get("anomaly_score")
        if pred_score is None:
            pred_score = torch.zeros(len(batch["image"]))
        return InferenceBatch(pred_score, outputs.get("pred_label"), outputs.get("pred_mask"), outputs.get("anomaly_map"))
    if isinstance(outputs, torch.Tensor):
        return InferenceBatch(outputs, None, None, None)
    raise TypeError(f"[ERROR] Unexpected model output: {type(outputs)}")
Patchcore.predict_step = patch_predict_step
Patchcore.validation_step = patch_predict_step

# --- Safe MVTec Dataset Patch ---
def safe_make_mvtec_ad_dataset(*args, **kwargs):
    root = kwargs.get("root") or args[0]
    category = kwargs.get("category") or args[1]
    split = kwargs.get("split") or args[2]
    extensions = kwargs.get("extensions") or ('.png', '.PNG')
    root_category = Path(root) / category
    def get_image_paths(folder):
        return sorted([str(p) for p in folder.glob("*") if p.suffix.lower() in extensions]) if folder.exists() else []
    train_good = get_image_paths(root_category / "train" / "good")
    test_good = get_image_paths(root_category / "test" / "good")
    test_anom = get_image_paths(root_category / "test" / "anomalous")
    gt_anom = get_image_paths(root_category / "ground_truth" / "anomalous")
    samples = []
    for img in train_good: samples.append({"image_path": img, "split": "train", "label_index": 0, "mask_path": None})
    for img in test_good: samples.append({"image_path": img, "split": "test", "label_index": 0, "mask_path": None})
    for img in test_anom: samples.append({"image_path": img, "split": "test", "label_index": 1, "mask_path": None})
    df = pd.DataFrame(samples)
    if len(test_anom) == len(gt_anom):
        mask_map = {Path(f).stem: f for f in gt_anom}
        for idx, row in df[(df.split == "test") & (df.label_index == 1)].iterrows():
            base = Path(row["image_path"]).stem
            df.at[idx, "mask_path"] = mask_map.get(base)
    df.attrs["task"] = "classification"
    return df
mvtec_module.make_mvtec_ad_dataset = safe_make_mvtec_ad_dataset

# --- Custom Image Transform ---
class ImageOnlyTransform:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    def __call__(self, image, mask=None):
        if isinstance(image, Image.Image):
            image = self.to_tensor(image)
        return self.normalize(image)

# --- Dataset & Loader ---
dataset_root = "c:/Users/auraosma/dataset"
category = "transistor"
image_transform = ImageOnlyTransform()
train_dataset = MVTecADDataset(root=dataset_root, category=category, split="train", augmentations=image_transform)
test_dataset = MVTecADDataset(root=dataset_root, category=category, split="test", augmentations=image_transform)
class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
def collate_fn(batch):
    images = [item.image for item in batch]
    masks = [getattr(item, "mask", None) for item in batch]
    return DotDict({"image": torch.stack(images), "gt_mask": masks})
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# --- Model & Training ---
model = Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3", "layer4"], pre_trained=False)
trainer = Trainer(max_epochs=1, accelerator="cpu", devices=1)
trainer.fit(model=model, train_dataloaders=train_loader)
torch.save(model.state_dict(), "patchcore_model.pth")
model.load_state_dict(torch.load("patchcore_model.pth"))
trainer.test(model=model, dataloaders=test_loader)


### Project 2: Google Vision OCR for Label Reading

# ocr_pipeline.py
import os
import csv
import io
from google.cloud import vision
from PIL import Image, ImageDraw

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "vision_key.json"

client = vision.ImageAnnotatorClient()

input_folder = "input_images"
output_csv = "ocr_output.csv"

def detect_text(path):
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description.strip() if texts else ""

def main():
    results = []
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            full_path = os.path.join(input_folder, filename)
            text = detect_text(full_path)
            results.append((filename, text))
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "OCR Text"])
        writer.writerows(results)

if __name__ == "__main__":
    main()
