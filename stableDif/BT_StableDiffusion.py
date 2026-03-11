import torch
import os
from huggingface_hub import login
from diffusers import StableDiffusionXLPipeline

login(token="xxx")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float32,
    use_safetensors=True
)

pipe.to("cpu")
prompt = "Photo of a (squirell: cow: 0.5)"
image = pipe(
    prompt=prompt, 
    num_inference_steps=2,
    guidance_scale=2.5,
    width=512,
    height=512
).images[0]

home = os.path.expanduser("~")
output_path = os.path.join(home, "Desktop/8. Semester/BachelorThesis")
if not os.path.exists(output_path):
    os.makedirs(output_path)

image.save(os.path.join(output_path, "hybrid1.png"))
print(f"ERFOLG! Das Bild wurde gespeichert: {output_path}/hybrid1.png")