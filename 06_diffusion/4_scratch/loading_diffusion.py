import os
import numpy
import torch
import torch
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


generator = torch.Generator("cuda").manual_seed(42)
prompt = 'a sleeping cat enjoying the sunshine.'
image = pipe(prompt, generator=generator).images[0]  # Generate image with a fixed seed
image.save(f"temp/sleeping_cat_seed.png")
pyplot.imshow(numpy.array(image))  # Convert PIL image to numpy array for display
pyplot.axis('off')  # Hide axes
pyplot.show()  # Display the image


prompt = 'a sleeping cat enjoing the sunshine.'
image = pipe(prompt, num_inference_steps=25).images[0]
image.save(f"temp/sleeping_cat_25.png")
pyplot.imshow(numpy.array(image))  # Convert PIL image to numpy array for display
pyplot.axis('off')  # Hide axes
pyplot.show()  # Display the image

