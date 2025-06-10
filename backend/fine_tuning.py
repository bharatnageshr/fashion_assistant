from transformers import CLIPModel, CLIPProcessor
from torch.optim import AdamW
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm


def custom_collate(batch):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    pixel_values = [item["pixel_values"] for item in batch]

    # Pad input_ids and attention_mask
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # Stack pixel values
    pixel_values = torch.stack(pixel_values)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values
    }

# Custom Dataset Class
class FashionCaptionDataset(Dataset):
    def __init__(self, image_paths, captions, processor):
        self.image_paths = image_paths
        self.captions = captions
        self.processor = processor

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        inputs = self.processor(
            text=[self.captions[idx]],
            images=image,
            return_tensors="pt",
            padding=True
        )
        # Squeeze batch dim (from processor) to match expected input
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "pixel_values": inputs["pixel_values"].squeeze(0)
        }

# Load your dataset (modify according to your structure)
def load_fashion_dataset():
    csv_path = 'archive/styles.csv'
    image_dir = 'archive/images'

    df = pd.read_csv(csv_path, quotechar='"', on_bad_lines='skip')
    df = df.dropna(subset=['id', 'productDisplayName'])

    image_paths = []
    captions = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        image_id = str(row['id'])
        caption = row['productDisplayName']
        image_path = os.path.join(image_dir, f"{image_id}.jpg")

        if os.path.exists(image_path):
            image_paths.append(image_path)
            captions.append(caption)

    return image_paths, captions

# Fine-tuning function
def fine_tune_clip(dataset, model, processor, epochs=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    loss_img = torch.nn.CrossEntropyLoss()
    loss_txt = torch.nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                return_loss=True
            )

            logits_per_image = outputs.logits_per_image
            logits_per_text = outputs.logits_per_text
            batch_size = logits_per_image.size(0)
            labels = torch.arange(batch_size, device=device)

            loss = (loss_img(logits_per_image, labels) +
                    loss_txt(logits_per_text, labels)) / 2

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    
    return model

# Main flow
if __name__ == "__main__":
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    print("Loading dataset...")
    image_paths, captions = load_fashion_dataset()
    dataset = FashionCaptionDataset(image_paths, captions, processor)

    print("Starting fine-tuning...")
    fine_tuned_model = fine_tune_clip(dataset, model, processor, epochs=3)

    print("Saving model...")
    fine_tuned_model.save_pretrained("./backend/fine_tuned_clip")
    processor.save_pretrained("./backend/fine_tuned_clip")
    print("Fine-tuning complete and model saved.")