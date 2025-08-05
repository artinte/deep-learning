import os
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from diffusers import StableDiffusionPipeline
from matplotlib import pyplot

assert torch.cuda.is_available(), "CUDA is not available. Please run on a machine with a GPU."

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=True
).to("cuda")

# Disable the safety checker for this example
def dummy_checker(images, **kwargs):
    return images, [False] * len(images)
pipe.safety_checker = dummy_checker

prompt = "a lovely cat running in the desert in Van Gogh style, trending art."
image = pipe(prompt).images[0]  # image here is in [PIL format](https://pillow.readthedocs.io/en/stable/)

# Now to display an image you can do either save it such as
os.makedirs("temp", exist_ok=True)
image.save(f"temp/lovely_cat.png")

pyplot.imshow(numpy.array(image))  # Convert PIL image to numpy array for display
pyplot.axis('off')  # Hide axes
pyplot.show()  # Display the image
