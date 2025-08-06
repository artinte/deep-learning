import itertools
import math
import numpy
import torch
import os
from diffusers import StableDiffusionPipeline
from matplotlib import pyplot

assert (
    torch.cuda.is_available()
), "CUDA is not available. Please run on a machine with a GPU."

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16",
    torch_dtype=torch.float16,
    use_auth_token=True,
).to("cuda")

os.makedirs("temp/diffprocess", exist_ok=True)

image_reservoir = []
latents_reservoir = []


@torch.no_grad()
def plot_show_callback(i, t, latents):
    latents_reservoir.append(latents.detach().cpu())
    image = pipe.vae.decode(1 / 0.18215 * latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]


@torch.no_grad()
def save_latents(i, t, latents):
    latents_reservoir.append(latents.detach().cpu())


@torch.no_grad()
def saveimg_callback(pipe, step_index, timestep, callback_kwargs):
    # This is the new function signature.

    # You can get the latents from the kwargs
    latents = callback_kwargs["latents"]

    # Add your existing logic here, perhaps with a frequency check
    if step_index % 10 == 0:
        # ... your existing image decoding and saving logic ...

        # Example of how to access and save the image
        image = pipe.vae.decode(1 / 0.18215 * latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image_np = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        pil_image = pipe.numpy_to_pil(image_np)[0]
        pil_image.save(f"temp/diffprocess/step_{step_index:04d}.png")

    # Important: The function must return the callback_kwargs
    return callback_kwargs


prompt = "a handsome cat dressed like Lincoln, trending art."
with torch.no_grad():
    image = pipe(
        prompt,
        # Use the callback parameter
        callback_on_step_end=saveimg_callback,
        # Tell the pipeline which tensors your callback needs access to
        callback_on_step_end_tensor_inputs=["latents"],
    ).images[0]

image.save(f"temp/lovely_cat_lincoln.png")
pyplot.imshow(numpy.array(image))
pyplot.axis("off")
pyplot.show()
