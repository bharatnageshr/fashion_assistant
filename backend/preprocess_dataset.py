# preprocess_dataset.py
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch
import faiss
import numpy as np
from transformers import CLIPProcessor, CLIPModel

# Paths
IMAGE_DIR = 'archive/images'
CSV_PATH = 'archive/styles.csv'
EMBEDDINGS_PATH = 'image_embeddings.npy'
INDEX_PATH = 'faiss_index.index'
ID_MAPPING_PATH = 'id_to_caption.json'

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load metadata
df = pd.read_csv(CSV_PATH, quotechar='"', on_bad_lines='skip')
df = df.dropna(subset=['id', 'productDisplayName'])

# Transform
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Store embeddings and ID->caption
image_embeddings = []
id_to_caption = {}

for _, row in tqdm(df.iterrows(), total=len(df)):
    image_id = str(row['id'])
    image_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
    if not os.path.exists(image_path):
        continue

    try:
        image = Image.open(image_path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            embedding = clip_model.get_image_features(**inputs)
        embedding = embedding.cpu().numpy().flatten()
        image_embeddings.append(embedding)

        # Compose caption
        caption = row['productDisplayName'] or ""
        id_to_caption[len(id_to_caption)] = {
            "id": image_id,
            "caption": caption
        }

    except Exception as e:
        print(f"Error processing {image_id}: {e}")

# Save embeddings and FAISS index
if not image_embeddings:
    print("No valid images found. Exiting.")
    exit()
image_embeddings_np = np.vstack(image_embeddings).astype('float32')
np.save(EMBEDDINGS_PATH, image_embeddings_np)

index = faiss.IndexFlatL2(image_embeddings_np.shape[1])
index.add(image_embeddings_np)
faiss.write_index(index, INDEX_PATH)

import json
with open(ID_MAPPING_PATH, 'w') as f:
    json.dump(id_to_caption, f)

print("Preprocessing complete.")