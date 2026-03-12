import replicate
import requests 
import os
import time

token = open("token.txt", "r")
os.environ["REPLICATE_API_TOKEN"] = token.read()

""" prompt = open("prompt.txt", "r") """
prompt_folder = os.path.expanduser("~/Desktop/8. Semester/BachelorThesis/replicate/prompt")
files_in_prompt = os.listdir(prompt_folder)
for prompts in files_in_prompt:
    if prompts.startswith('.'): continue
    prompt_path = os.path.join(prompt_folder, prompts)
    with open(prompt_path, "r") as f:
        current_prompt = f.read()



    output = replicate.run(
        "black-forest-labs/flux-dev",
        input={
            "prompt": current_prompt,
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
    count = 1
    folder_path = os.path.expanduser("~/Desktop/8. Semester/BachelorThesis/replicate/images/squirrel_cow")
    files_in_folder = os.listdir(folder_path)
    for i in files_in_folder:
        full_path = os.path.join(folder_path, i)
        if os.path.isfile(full_path):
            count += 1
    print(count)

    image_url = str(output[0])
    print(f"Bild-URL: {image_url}")

    timestamp = int(time.time()) 
    img_data = requests.get(image_url).content
    save_path = os.path.expanduser(f"~/Desktop/8. Semester/BachelorThesis/replicate/images/squirrel_cow/hybrid_{count}.png")

    with open(save_path, 'wb') as handler:
        handler.write(img_data)
    time.sleep(10)
    print("pause")
