import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import shutil
from tqdm import tqdm
from typing import Any
import matplotlib.pyplot as plt
import matplotlib

from dataclasses import dataclass

from enum import Enum, IntEnum, auto


class MaskType(Enum):
    CAR = auto()
    REAR_VIEW_MIRROR = auto()
    LICENSE_PLATE = auto()
    CAR_WHEEL = auto()

    def get_mask_folder(self, masks_folder: str) -> str:
        """aaaaaaaa"""
        return os.path.join(masks_folder, self.name.lower())

    @property
    def prompt_name(self) -> str:
        return self.name.lower().replace("_", " ")

    @staticmethod
    def from_prompt_name(name: str) -> "MaskType":
        return MaskType[name.upper().replace(" ", "_")]


def clean_folder(folder: str) -> None:
    if os.path.isdir(folder):
        shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder, exist_ok=True)


@dataclass
class DetectionResults:
    masks: dict[MaskType, np.ndarray]
    color_mask: np.ndarray


def run_detection(img_name: str,
                  image_folder: str,
                  sam2_predictor: SAM2ImagePredictor,
                  grounding_model: Any,
                  mask_types: list[MaskType]) -> DetectionResults:
    """aaaaa

    return dict of binary mask per mask type, and color mask aggregating all masks superimposed on the image"""
    img_path = os.path.join(image_folder, img_name)
    image_source, image = load_image(img_path)
    sam2_predictor.set_image(image_source)

    mask_names = [mask.prompt_name for mask in mask_types]
    BOX_THRESHOLD = 0.35
    TEXT_THRESHOLD = 0.25
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=" ".join([n+"." for n in mask_names]),
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
    )

    # process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])

    # Split into car and non-car boxes
    car_boxes, car_confidences, car_labels = [], [], []
    other_boxes, other_confidences, other_labels = [], [], []
    for box, conf, label in zip(boxes, confidences, labels):
        if label == MaskType.CAR.prompt_name:
            car_boxes.append(box)
            car_confidences.append(conf)
            car_labels.append(label)
        else:
            other_boxes.append(box)
            other_confidences.append(conf)
            other_labels.append(label)

    # Keep only the largest box for MaskType.CAR
    if len(car_boxes) > 0:
        largest_area_idx = np.argmax([box[-2]*box[-1] for box in car_boxes])
        boxes = torch.stack(other_boxes+[car_boxes[largest_area_idx]])
        confidences = torch.stack(
            other_confidences+[car_confidences[largest_area_idx]])
        labels = other_labels + [car_labels[largest_area_idx]]

    input_boxes = box_convert(
        boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # -------------------------------- vvv -----------------------------------
    # warning:
    # https://github.com/IDEA-Research/Grounded-SAM-2/issues/38
    # changing bloat16 to float16
    torch.autocast(device_type="cuda", dtype=torch.float16).__enter__()
    # -------------------------------- ^^^ -----------------------------------

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    try:
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
            return_logits=False,
        )
    except:
        print(
            f"Error during SAM2 prediction for image {img_name} in {image_folder}")
        img = cv2.imread(img_path)
        return DetectionResults({}, img)

    # convert the shape to (n, H, W)
    if masks.ndim == 4:
        masks = masks.squeeze(1)

    """
    Visualize image with supervision useful API
    """
    img = cv2.imread(img_path)
    detections = sv.Detections(
        xyxy=input_boxes,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
        class_id=np.arange(len(labels))
    )

    box_annotator = sv.BoxAnnotator()
    color_mask = box_annotator.annotate(
        scene=img.copy(), detections=detections)
    label_annotator = sv.LabelAnnotator()
    color_mask = label_annotator.annotate(scene=color_mask, detections=detections,
                                          labels=[
                                              f"{label} {confidence:.2f}"
                                              for label, confidence
                                              in zip(labels, confidences.numpy().tolist())
                                          ])
    mask_annotator = sv.MaskAnnotator()
    color_mask = mask_annotator.annotate(
        scene=color_mask, detections=detections)

    # Create binary masks for each mask type
    binary_masks_per_type = {mask_type: np.zeros(
        masks[0].shape, dtype=bool) for mask_type in mask_types}

    for mask, label in zip(masks, labels):
        try:
            t = MaskType.from_prompt_name(label)
            binary_masks_per_type[t] |= mask.astype(bool)
        except:
            print(
                f"Warning: label '{label}' not in MaskType enum. Skipping this mask ({img_name}).")
            # See https://github.com/IDEA-Research/Grounded-SAM-2/issues/50
            # See https://github.com/IDEA-Research/Grounded-SAM-2/issues/66
            # See https://github.com/IDEA-Research/Grounded-SAM-2/issues/67
            continue

    return DetectionResults(binary_masks_per_type, color_mask)


def main(image_folder: str,
         out_masks_folder: str,
         out_color_masks_folder: str,
         mask_types: list[MaskType]) -> None:
    """aaaaa"""
    SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
    SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth"

    assert torch.cuda.is_available()
    DEVICE = "cuda"

    # Build SAM2 image predictor
    sam2_checkpoint = SAM2_CHECKPOINT
    model_cfg = SAM2_MODEL_CONFIG
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)

    # build grounding dino model
    grounding_model = load_model(model_config_path=GROUNDING_DINO_CONFIG,
                                 model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
                                 device=DEVICE)

    # Iterate through the images in the image_folder
    clean_folder(out_masks_folder)
    clean_folder(out_color_masks_folder)

    image_names = sorted([f.name for f in os.scandir(
        image_folder) if f.name.endswith('.png')])

    for img_name in tqdm(image_names, desc="Grounded-SAM 2", unit="img"):
        torch.cuda.empty_cache()
        results = run_detection(img_name, image_folder,
                                sam2_predictor, grounding_model, mask_types)

        cv2.imwrite(os.path.join(out_color_masks_folder,
                    img_name), results.color_mask)
        for mask_type, binary_mask in results.masks.items():
            out_mask_folder = mask_type.get_mask_folder(out_masks_folder)
            os.makedirs(out_mask_folder, exist_ok=True)
            cv2.imwrite(os.path.join(out_mask_folder, img_name),
                        binary_mask.astype(np.uint8)*255)


def get_time_str() -> str:
    """Get the current time in Europe/Paris."""
    return datetime.now(ZoneInfo("Europe/Paris")).strftime("[%Y/%m/%d - %H:%M:%S] ")


SHARED_DATA_SHOWROOM_DIR = "../shared_data/showroom"


def run(list_project_id: list[str] | None, car: bool) -> None:
    main_dir = os.path.join(SHARED_DATA_SHOWROOM_DIR,
                            "car" if car else "object")

    archives_folder = os.path.join(main_dir, "archives")
    processing_folder = os.path.join(main_dir, "processing")

    if list_project_id is None:
        list_project_id = sorted([os.path.splitext(f.name)[0]
                                 for f in os.scandir(archives_folder) if f.is_file()])

    mask_error_txt = os.path.join(processing_folder, "mask_errors.txt")
    mask_success_txt = os.path.join(processing_folder, "mask_success.txt")

    for project_id in list_project_id:
        named_folder = os.path.join(processing_folder, project_id)

        try:
            with open(mask_success_txt, "a") as f:
                f.write(get_time_str() + f"Start: {project_id}\n")

            stats_json = f"{named_folder}/masks/stats.json"
            print(f"Processing {project_id}...")

            image_folder = f"{named_folder}/preprocessing/centered_undist_images"
            if os.path.exists(stats_json):
                print("Already processed, skipping...")
            elif not os.path.exists(image_folder):
                print(f"Image folder not found: {image_folder}")
            else:
                start = time.perf_counter()
                main(image_folder=image_folder,
                     out_masks_folder=f"{named_folder}/masks",
                     out_color_masks_folder=f"{named_folder}/masks/cache",
                     mask_types=list(MaskType))
                end = time.perf_counter()

                with open(stats_json, "w", encoding="utf-8") as _f:
                    json.dump({"global_compute_time_s": end-start},
                              _f, indent=4)

            with open(mask_success_txt, "a") as f:
                f.write(get_time_str() + f"Success: {project_id}\n")

        except Exception:
            with open(mask_error_txt, "a") as f:
                f.write(get_time_str() + f"Exception {project_id}:\n")
                f.write(traceback.format_exc())  # Get full traceback
                f.write("\n" + "-" * 50 + "\n")  # Separator for readability


if __name__ == "__main__":
    # list_project_id = [
    #     # "9b91575e-6a07-4401-8437-c6d320b06942",
    #     "3d292b8a-f557-4df7-adb4-a8a92746c796",
    #     # "54e04ed0-a99a-4399-af47-e4bbb1970424"
    # ]
    list_project_id = None
    run(list_project_id, car=True)
