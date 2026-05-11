from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import os

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

folder_path = "/Users/eldarakhundzada/Desktop/8. Semester/BachelorThesis/DeepFace/TestIMG"
#image = Image.open(requests.get(folder_path, stream=True).raw)
image_path = os.path.expanduser(folder_path)
image_list = os.listdir(image_path)

for filename in image_list:
    if(filename).startswith("."): continue
    full_image_path = os.path.join(image_path, filename)

    current_image = Image.open(full_image_path)
    #labels = ["not ai generated", "ai generated"]
    labels = ["is facemorphing", "is not facemorphing"]

    inputs = processor(text = labels, images=current_image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)

    print("\n" + "="*30)
    print(f"ANALYSE FÜR: {folder_path}")
    print("="*30)

    for i, label in enumerate(labels):
        percentage = probs[0][i].item() * 100
        print(f"{label:30} : {percentage:>6.2f}%")
    print("="*30)
