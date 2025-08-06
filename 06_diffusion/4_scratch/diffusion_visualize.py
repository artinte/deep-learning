import functools
import itertools
import math
import mediapy
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
def save_latents(i, t, latents):
    latents_reservoir.append(latents.detach().cpu())


@torch.no_grad()
def saveimg_callback(pipe, step_index, timestep, callback_kwargs, frequency=10):
    # You can get the latents from the kwargs
    latents = callback_kwargs["latents"]

    # Add your existing logic here, perhaps with a frequency check
    if step_index % frequency == 0:
        # Example of how to access and save the image
        image = pipe.vae.decode(1 / 0.18215 * latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image_np = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        pil_image = pipe.numpy_to_pil(image_np)[0]
        pil_image.save(f"temp/diffprocess/step_{step_index:04d}_{timestep}.png")
        image_reservoir.append(image_np)
        latents_reservoir.append(latents.detach().cpu())
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


image_reservoir = []
latents_reservoir = []
prompt = "a lovely cat running in the desert in Van Gogh style, trending art."
with torch.no_grad():
    image = pipe(
        prompt,
        callback_on_step_end=functools.partial(saveimg_callback, frequency=1),
        callback_on_step_end_tensor_inputs=["latents"],
    ).images[0]
image.save(f"temp/lovely_cat_vangogh.png")
mediapy.write_video("temp/lavely_cat_vangopy.mp4", image_reservoir, fps=10)


print('Lantents shape:', latents_reservoir[0].shape)
latents_np_seq = [tsr[0, [0, 1, 2]].permute(1, 2, 0).numpy() for tsr in latents_reservoir]
mediapy.write_video("temp/latents_seq.mp4", latents_np_seq, fps=10)

