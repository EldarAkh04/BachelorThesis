from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image_path = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_path, stream=True).raw)

labels = ["a photo of a cat", "a photo of a dog"]

inputs = processor(text = labels, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

print("\n" + "="*30)
print(f"ANALYSE FÜR: {image_path}")
print("="*30)

for i, label in enumerate(labels):
    percentage = probs[0][i].item() * 100
    print(f"{label:30} : {percentage:>6.2f}%")
print("="*30)
