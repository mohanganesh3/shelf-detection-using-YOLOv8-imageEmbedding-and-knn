import os
import glob
import re
from ultralytics import YOLO
from src.img2vec_resnet18 import Img2VecResnet18
from PIL import Image, ImageEnhance
from sklearn.neighbors import NearestNeighbors
from collections import Counter
from pathlib import Path
import torch.serialization
import numpy as np
import yaml
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_PATH = config["paths"]["model"]
DATA_PATH = config["paths"]["data"]
KNOWLEDGE_BASE_PATH = config["paths"]["knowledge_base"]
N_NEIGHBORS = config["thresholds"]["knn_neighbors"]
YOLO_CONF = config["thresholds"]["yolo_conf"]
YOLO_IOU = config["thresholds"]["yolo_iou"]

def get_latest_yolo_dir(base_path, stem):
    """Find the latest YOLO output directory matching the input stem."""
    pattern = os.path.join(base_path, f"{stem}*")
    dirs = glob.glob(pattern)
    if not dirs:
        logger.warning(f"No YOLO output directories found for stem {stem}")
        return None
    dirs.sort(key=lambda x: (int(re.search(r'\d+$', x).group()) if re.search(r'\d+$', x) else 0, x))
    return dirs[-1]

def process_image(image_path):
    """Process an image to identify and count products."""
    logger.info(f"Processing image: {image_path}")
    
    if not os.path.exists(image_path):
        logger.error(f"Input file {image_path} does not exist")
        return {"error": f"Input file {image_path} does not exist."}, None, None, None

    # Preprocess image
    try:
        img = Image.open(image_path)
        img = ImageEnhance.Contrast(img).enhance(1.2)
        temp_path = os.path.join(config["paths"]["temp_image"], Path(image_path).stem + "_processed" + Path(image_path).suffix)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        img.save(temp_path)
        logger.info(f"Image preprocessed and saved to {temp_path}")
    except Exception as e:
        logger.error(f"Failed to preprocess image: {e}")
        return {"error": f"Failed to preprocess image: {e}"}, None, None, None

    # Allowlist Ultralytics class
    #torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])

    # Load YOLO model
    try:
        model = YOLO(MODEL_PATH)
        logger.info(f"YOLO model loaded from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return {"error": f"Failed to load YOLO model: {e}"}, None, None, None

    # Extract image stem
    PATH = Path(image_path).stem

    # Run YOLO detection
    try:
        results = model.predict(
            source=temp_path,
            save=True,
            save_crop=True,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            project=DATA_PATH,
            name=PATH,
        )
        logger.info(f"YOLO prediction completed for {image_path}")
    except Exception as e:
        logger.error(f"YOLO prediction failed: {e}")
        return {"error": f"YOLO prediction failed: {e}"}, None, None, None

    # Find latest YOLO output directory
    output_dir = get_latest_yolo_dir(DATA_PATH, PATH)
    if not output_dir:
        logger.error(f"No YOLO output directory found for stem {PATH}")
        return {"error": f"No YOLO output directory found for stem '{PATH}'"}, None, None, None

    # Extract bounding boxes
    annotations = []
    result = results[0]
    if result.boxes:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for box, conf in zip(boxes, confs):
            annotations.append({
                'box': box.tolist(),
                'conf': float(conf),
                'label': None,
                'product': None
            })

    # Get knowledge base images
    list_imgs = glob.glob(f"{KNOWLEDGE_BASE_PATH}/**/*.jpg", recursive=True)
    if not list_imgs:
        logger.warning(f"No images found in knowledge base at {KNOWLEDGE_BASE_PATH}")
        return {"error": f"No images found in knowledge base"}, None, None, annotations

    # Initialize Img2VecResnet18
    try:
        img2vec = Img2VecResnet18()
        logger.info("Img2VecResnet18 initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Img2VecResnet18: {e}")
        return {"error": f"Failed to initialize Img2VecResnet18: {e}"}, None, None, annotations

    # Extract feature vectors
    classes = []
    embeddings = []
    for filename in list_imgs:
        try:
            I = Image.open(filename)
            vec = img2vec.getVec(I)
            I.close()
            if np.var(vec) < 1e-6:
                logger.warning(f"Low-variance embedding for {filename}")
                continue
            folder_path = os.path.dirname(filename)
            folder_name = os.path.basename(folder_path)
            classes.append(folder_name)
            embeddings.append(vec)
        except Exception as e:
            logger.warning(f"Failed to process knowledge base image {filename}: {e}")

    if not embeddings:
        logger.error("No valid embeddings extracted from knowledge base")
        return {"error": "No valid embeddings extracted from knowledge base"}, None, None, annotations

    # Fit k-NN model
    try:
        model_knn = NearestNeighbors(metric='cosine', n_neighbors=min(N_NEIGHBORS, len(embeddings)))
        model_knn.fit(embeddings)
        logger.info("k-NN model fitted")
    except Exception as e:
        logger.error(f"Failed to fit k-NN model: {e}")
        return {"error": f"Failed to fit k-NN model: {e}"}, None, None, annotations

    # Process cropped images
    list_crops = glob.glob(f"{output_dir}/crops/object/*.jpg")
    if not list_crops:
        logger.warning(f"No cropped images found in {output_dir}/crops/object/")
        return {"error": f"No cropped images found in {output_dir}/crops/object/"}, None, None, annotations

    predictions = []
    for i, IMG_DIR in enumerate(list_crops):
        try:
            I = Image.open(IMG_DIR)
            vec = img2vec.getVec(I)
            I.close()
            
            dists, idx = model_knn.kneighbors([vec])
            weights = 1 / (dists[0] + 1e-6)
            brands_nearest_neighbors = [classes[i] for i in idx[0]]
            count = Counter()
            for brand, weight in zip(brands_nearest_neighbors, weights):
                count[brand] += weight
                
            product = "unknown"
            prob = 0.0
            if count:
                product, score = sorted(count.items(), key=lambda item: item[1])[-1]
                prob = score / sum(weights)
                
            crop_name = Path(IMG_DIR).stem
            predictions.append((crop_name, product, prob, IMG_DIR))
            
            annotation_idx = i  # Use index directly since crop names may not have reliable numbering
            if annotation_idx < len(annotations):
                annotations[annotation_idx]['label'] = f"{product.replace('_', ' ')} ({prob:.0%})"
                annotations[annotation_idx]['product'] = product
                
        except Exception as e:
            logger.warning(f"Failed to process cropped image {IMG_DIR}: {e}")

    # Count products
    product_counts = Counter(pred[1] for pred in predictions if pred[2] > config["thresholds"]["knn_conf"])

    # Write to file
    file_path = f"{output_dir}/predictions.txt"
    try:
        with open(file_path, "w") as file:
            file.write("Individual Crop Predictions:\n")
            for crop_name, product, prob, _ in predictions:
                file.write(f"The crop image {crop_name} is predicted as {product.replace('_', ' ')} with {prob:.0%} probability\n")
            file.write("\nSummary of Product Counts:\n")
            if product_counts:
                for product, count in sorted(product_counts.items()):
                    file.write(f"Found {count} {product.replace('_', ' ')}(s)\n")
            else:
                file.write("No products identified with high confidence.\n")
                
            file.write(f"\nTotal object detections: {len(annotations)}\n")
            file.write(f"Objects with identified products: {len([p for p in predictions if p[2] > config['thresholds']['knn_conf']])}\n")
            file.write(f"Objects with uncertain classification: {len([p for p in predictions if p[2] <= config['thresholds']['knn_conf']])}\n")
        logger.info(f"Predictions written to {file_path}")
    except Exception as e:
        logger.error(f"Failed to write to {file_path}: {e}")
        return {"error": f"Failed to write to {file_path}: {e}"}, None, None, annotations

    logger.info("Image processing completed successfully")
    return None, predictions, product_counts, annotations

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Product Identification and Counting')
    parser.add_argument('--input', help='Input file path', required=True)
    args = parser.parse_args()

    error, predictions, product_counts, annotations = process_image(args.input)
    if error:
        print(f"Error: {error['error']}")
        exit(1)

    print("Summary of Product Counts:")
    if product_counts:
        for product, count in sorted(product_counts.items()):
            print(f"Found {count} {product.replace('_', ' ')}(s)")
    else:
        print("No products identified with high confidence.")