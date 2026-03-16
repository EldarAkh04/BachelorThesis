from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import os

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image_path = "~/Desktop/8. Semester/BachelorThesis/replicate/images/squirrel_cow/hybrid_1.png"
#image = Image.open(requests.get(image_path, stream=True).raw)
image = os.path.expanduser(image_path)

labels = ["a cat","a dog"]

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
