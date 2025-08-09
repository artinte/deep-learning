import os
import numpy
import torch
import torch
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


def pipe_simplified(
    prompt=["a lovely cat"],
    negative_prompt=[""],
    # `num_inference_steps` is the number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
    # It is recommended to use between 50 and 150 denoising steps, with 50 being a good default.
    num_inference_steps=50,
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    # Higher guidance scale encourages to generate images closely linked to the text `prompt`,
    # usually at the expense of lower image quality.
    # Guidance scale of 7.5 is a good default value.
    # Guidance scale of 1.0 is equivalent to doing no classifier free guidance.
    guidance_scale=7.5,
):

    batch_size = 1
    height, width = 512, 512
    generator = None

    # get prompt text embeddings
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    text_embeddings = pipe.text_encoder(text_input_ids.to(pipe.device))[0]
    bs_embed, seq_len, _ = text_embeddings.shape
    print(f"Text embeddings shape: {text_embeddings.shape}")

    # get negative prompts  text embedding
    max_length = text_input_ids.shape[-1]
    uncond_input = pipe.tokenizer(
        negative_prompt,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(pipe.device))[0]
    print(f"Unconditional text embeddings shape: {uncond_embeddings.shape}")

    # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
    seq_len = uncond_embeddings.shape[1]
    uncond_embeddings = uncond_embeddings.repeat(batch_size, 1, 1)
    uncond_embeddings = uncond_embeddings.view(batch_size, seq_len, -1)

    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    print(f"Concatenated text embeddings shape: {text_embeddings.shape}")

    # get the initial random noise unless the user supplied it
    # Unlike in other pipelines, latents need to be generated in the target device
    # for 1-to-1 results reproducibility with the CompVis implementation.
    latents_shape = (batch_size, pipe.unet.in_channels, height // 8, width // 8)
    latents_dtype = text_embeddings.dtype
    latents = torch.randn(
        latents_shape, generator=generator, device=pipe.device, dtype=latents_dtype
    )

    # set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps)
    # Some schedulers like PNDM have timesteps as arrays
    # It's more optimized to move all timesteps to correct device beforehand
    timesteps_tensor = pipe.scheduler.timesteps.to(pipe.device)
    # scale the initial noise by the standard deviation required by the scheduler
    latents = latents * pipe.scheduler.init_noise_sigma

    # Main diffusion process
    for i, t in enumerate(pipe.progress_bar(timesteps_tensor)):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        # predict the noise residual
        noise_pred = pipe.unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(
            noise_pred,
            t,
            latents,
        ).prev_sample

    latents = 1 / 0.18215 * latents
    image = pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
    image = image.detect().cpu().permute(0, 2, 3, 1).float().numpy()
    return image


image = pipe_simplified(
    prompt=["a lovely cat"],
    negative_prompt=["Sunshine"],
)
# The two lines below are the problem. Remove them.
image = pipe.numpy_to_pil(image)[0]
image.save(f"temp/lovely_cat_simplified.png")
pyplot.imshow(numpy.array(image))
pyplot.axis("off")
pyplot.show()


image = pipe_simplified(
    prompt = ["a cat dressed like a ballerina"],
    negative_prompt = [""],)
# The two lines below are the problem. Remove them.
image = pipe.numpy_to_pil(image)[0]
image.save(f"temp/dressed_car_simplified.png")
pyplot.imshow(numpy.array(image))
pyplot.axis("off")
pyplot.show()