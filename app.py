import streamlit as st
import os
import shutil
from PIL import Image, ImageDraw, ImageFont
from main import process_image
from pathlib import Path
import yaml
import uuid

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

DATA_PATH = config["paths"]["data"]
KNOWLEDGE_BASE_PATH = config["paths"]["knowledge_base"]
COLOR_MAP = config["colors"]
KNN_CONF_THRESHOLD = config["thresholds"]["knn_conf"]

st.title("Product Identification and Counting System")

# Initialize session state
if 'unknown_images' not in st.session_state:
    st.session_state.unknown_images = {}
if 'confirmed_classes' not in st.session_state:
    st.session_state.confirmed_classes = {}

def draw_annotations(image_path, annotations):
    """Draw bounding boxes and labels on the image."""
    try:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        for ann in annotations:
            box = ann['box']
            label = ann.get('label')
            color = COLOR_MAP.get(ann.get('product', 'unknown'), 'white')

            # Draw box
            draw.rectangle(box, outline=color, width=2)
            
            # Draw label if it exists
            if label:
                text_bbox = draw.textbbox((box[0], box[1] - 20), label, font=font)
                draw.rectangle((text_bbox[0], text_bbox[1], text_bbox[2], text_bbox[3]), fill=color)
                draw.text((box[0], box[1] - 20), label, fill='black', font=font)

        return image
    except Exception as e:
        st.error(f"Error drawing annotations: {e}")
        return Image.open(image_path)

def save_to_knowledge_base(crop_path, product_class):
    """Save a crop image to the knowledge base directory with a unique filename."""
    target_dir = os.path.join(KNOWLEDGE_BASE_PATH, product_class)
    os.makedirs(target_dir, exist_ok=True)
    
    # Generate a unique filename using UUID
    filename = f"{product_class}_{str(uuid.uuid4())[:8]}.jpg"
    target_path = os.path.join(target_dir, filename)
    
    try:
        shutil.copy(crop_path, target_path)
        st.success(f"Saved {filename} to knowledge base as {product_class}")
        return target_path
    except Exception as e:
        st.error(f"Error saving to knowledge base: {e}")
        return None

# File uploader
uploaded_file = st.file_uploader("Upload an image (e.g., retail shelf)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file
    image_path = os.path.join(DATA_PATH, "img", uploaded_file.name)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    try:
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except Exception as e:
        st.error(f"Error saving uploaded image: {e}")
        st.stop()
    
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing image...")

    # Process image
    error, predictions, product_counts, annotations = process_image(image_path)

    if error:
        st.error(f"Error: {error['error']}")
        st.stop()

    # Display annotated image
    st.subheader("Annotated Image")
    annotated_image = draw_annotations(image_path, annotations)
    st.image(annotated_image, caption="Image with Product Annotations", use_column_width=True)

    # Get product classes from knowledge base
    product_classes = [d for d in os.listdir(KNOWLEDGE_BASE_PATH) if os.path.isdir(os.path.join(KNOWLEDGE_BASE_PATH, d))]
    
    # Process predictions
    unknown_crops = []
    known_predictions = []
    
    if predictions:
        for i, (crop_name, product, prob, crop_path) in enumerate(predictions):
            if prob < KNN_CONF_THRESHOLD or product == "unknown":
                unknown_crops.append((crop_name, product, prob, crop_path))
                st.session_state.unknown_images[crop_path] = {"name": crop_name, "suggested": product, "prob": prob}
            else:
                known_predictions.append((crop_name, product, prob, crop_path))
    st.subheader("Summary of Product Counts")
    if product_counts:
        for product, count in sorted(product_counts.items()):
            st.write(f"Found {count} {product.replace('_', ' ')}(s)")
    else:
        st.write("No products identified.")

    # Display confident predictions
    st.subheader("Confident Predictions")
    max_crops = 20
    if known_predictions:
        for i, (crop_name, product, prob, crop_path) in enumerate(known_predictions):
            if i >= max_crops:
                st.write(f"Showing first {max_crops} of {len(known_predictions)} predictions.")
                break
            col1, col2 = st.columns([1, 3])
            with col1:
                try:
                    crop_img = Image.open(crop_path)
                    st.image(crop_img, caption=crop_name, width=100)
                except:
                    st.write(f"(Image {crop_name} not found)")
            with col2:
                st.write(f"The crop image {crop_name} is predicted as {product.replace('_', ' ')} with {prob:.0%} probability")
    else:
        st.write("No confident predictions available.")
    
    # Display and confirm uncertain predictions
    if unknown_crops:
        st.subheader("Uncertain or Unknown Products")
        st.write("Please confirm the product type for these uncertain predictions:")
        
        for crop_path, data in list(st.session_state.unknown_images.items()):
            if crop_path in st.session_state.confirmed_classes:
                continue
                
            crop_name = data["name"]
            suggested = data["suggested"]
            prob = data["prob"]
            
            try:
                col1, col2, col3 = st.columns([1, 2, 2])
                with col1:
                    crop_img = Image.open(crop_path)
                    st.image(crop_img, caption=crop_name, width=100)
                
                with col2:
                    st.write(f"Suggested: {suggested.replace('_', ' ')} ({prob:.0%})")
                
                with col3:
                    product_options = ["Select class..."] + ["NEW BRAND"] + product_classes
                    product_selection = st.selectbox(
                        f"Confirm product type for {crop_name}",
                        options=product_options,
                        key=f"select_{crop_path}"
                    )
                    
                    new_brand_name = None
                    if product_selection == "NEW BRAND":
                        new_brand_name = st.text_input(
                            "Enter new brand name (lowercase, use underscores for spaces):",
                            key=f"new_brand_{crop_path}"
                        )
                    
                    if st.button("Confirm", key=f"confirm_{crop_path}"):
                        if product_selection == "NEW BRAND" and new_brand_name:
                            new_brand_name = new_brand_name.strip().lower().replace(" ", "_")
                            if new_brand_name and new_brand_name not in product_classes:
                                save_to_knowledge_base(crop_path, new_brand_name)
                                st.session_state.confirmed_classes[crop_path] = new_brand_name
                                del st.session_state.unknown_images[crop_path]
                                st.rerun()
                        elif product_selection != "Select class...":
                            save_to_knowledge_base(crop_path, product_selection)
                            st.session_state.confirmed_classes[crop_path] = product_selection
                            del st.session_state.unknown_images[crop_path]
                            st.rerun()
            except Exception as e:
                st.error(f"Error processing image {crop_name}: {e}")
    
    # Display product counts
    
    
            
    # Clean up session state
    if st.button("Reset Confirmations"):
        st.session_state.unknown_images = {}
        st.session_state.confirmed_classes = {}
        st.success("All confirmations reset!")