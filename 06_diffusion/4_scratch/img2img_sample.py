from io import BytesIO
from PIL import Image
import os
import requests
import torch
from matplotlib import pyplot
from diffusers import StableDiffusionImg2ImgPipeline

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"


os.makedirs("temp", exist_ok=True)
response = requests.get(url)
init_img = Image.open(BytesIO(response.content)).convert("RGB")
init_img = init_img.resize((768, 512))
init_img.save("temp/sketch-mountains-input.jpg")
pyplot.imshow(init_img)
pyplot.axis("off")
pyplot.show()

model_path = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_path, revision="fp16", torch_dtype=torch.float16, use_auth_token=True
)
pipe = pipe.to(device)

prompt = "A fantasy landscape, trending on artstation"
generator = torch.Generator(device=device).manual_seed(1024)
image = pipe(
    prompt=prompt,
    image=init_img,
    strength=0.75,
    num_inference_steps=50,
    guidance_scale=7.5,
    generator=generator,
).images[0]

image.save("temp/fantasy_landscape.png")
pyplot.imshow(image)
pyplot.axis("off")
pyplot.show()
