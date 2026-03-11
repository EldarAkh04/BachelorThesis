import replicate
import requests 
import os
import time

token = open("token.txt", "r")
os.environ["REPLICATE_API_TOKEN"] = token.read()

output = replicate.run(
    "black-forest-labs/flux-dev",
    input={
        "prompt": "A bizarre biological hybrid: the body, bushy tail, and furry red squirrel legs, with the paws transitioning into miniature cloven cow hooves. It has the head, horns, typical cow nose, and white-and-brown patched fur pattern of a cow. The creature is squirrel-sized, sitting on a forest branch. Realistic taxidermy style, 8k, highly detailed.",
        "go_fast": True,
        "guidance": 3.5,
        "megapixels": "1",
        "num_outputs": 1,
        "aspect_ratio": "1:1",
        "output_format": "png",
        "output_quality": 90,
        "num_inference_steps": 35
    }
)

image_url = str(output[0])
print(f"Bild-URL: {image_url}")

timestamp = int(time.time()) 
img_data = requests.get(image_url).content
save_path = os.path.expanduser(f"~/Desktop/8. Semester/BachelorThesis/replicate/hybrid_{timestamp}.png")

# Sicherstellen, dass der Ordner existiert
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, 'wb') as handler:
    handler.write(img_data)

print(f"ERFOLG! Das Bild wurde hier gespeichert: {save_path}")