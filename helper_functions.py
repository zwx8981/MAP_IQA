"""Diffusion math and image utilities; importing this module does not load models."""

import torch
from PIL import Image, ImageOps


# alpha and beta for DDIM
def get_alpha_and_beta(t, scheduler):
    # want to run this for both current and previous timnestep
    if t < 0:
        return scheduler.final_alpha_cumprod.item(), 1 - scheduler.final_alpha_cumprod.item()

    if t.dtype == torch.long or (t == t.long()):
        alpha = scheduler.alphas_cumprod[t.cpu().long()]
        return alpha.item(), 1 - alpha.item()

    low = t.floor().long()
    high = t.ceil().long()
    rem = t - low

    low_alpha = scheduler.alphas_cumprod[low]
    high_alpha = scheduler.alphas_cumprod[high]
    interpolated_alpha = low_alpha * rem + high_alpha * (1 - rem)
    interpolated_beta = 1 - interpolated_alpha
    return interpolated_alpha.item(), interpolated_beta.item()

#forward DDIM step
def forward_step(
        self,
        model_output,
        timestep: int,
        sample,
):
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    prev_timestep = timestep - self.config.num_train_timesteps / self.num_inference_steps

    if timestep > self.timesteps.max():
        raise NotImplementedError("Need to double check what the overflow is")

    alpha_prod_t, beta_prod_t = get_alpha_and_beta(timestep, self)
    alpha_prod_t_prev, _ = get_alpha_and_beta(prev_timestep, self)

    alpha_quotient = ((alpha_prod_t / alpha_prod_t_prev) ** 0.5)
    first_term = (1. / alpha_quotient) * sample
    second_term = (1. / alpha_quotient) * (beta_prod_t ** 0.5) * model_output
    third_term = ((1 - alpha_prod_t_prev) ** 0.5) * model_output

    return first_term - second_term + third_term


# reverse ddim step
def reverse_step(
        self,
        model_output,
        timestep: int,
        sample,
):
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
        )

    prev_timestep = timestep - self.config.num_train_timesteps / self.num_inference_steps

    if timestep > self.timesteps.max():
        raise NotImplementedError
    else:
        alpha_prod_t = self.alphas_cumprod[timestep]

    alpha_prod_t, beta_prod_t = get_alpha_and_beta(timestep, self)
    alpha_prod_t_prev, _ = get_alpha_and_beta(prev_timestep, self)

    alpha_quotient = ((alpha_prod_t / alpha_prod_t_prev) ** 0.5)

    first_term = alpha_quotient * sample
    second_term = ((beta_prod_t) ** 0.5) * model_output
    third_term = alpha_quotient * ((1 - alpha_prod_t_prev) ** 0.5) * model_output
    return first_term + second_term - third_term


def pad_to_multiple_of_64(image):
    """Return the padded PIL image and its borders, with both dimensions divisible by 64."""
    width, height = image.size

    # Compute the total padding along each axis.
    pad_h = (64 - height % 64) % 64
    pad_w = (64 - width % 64) % 64

    # Split the padding between opposite sides to keep the image centered.
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Fill the borders with black pixels.
    padded_image = ImageOps.expand(
        image,
        border=(pad_left, pad_top, pad_right, pad_bottom),
        fill=0
    )

    return padded_image, (pad_left, pad_top, pad_right, pad_bottom)


def crop_to_original(image, padding):
    """Remove the recorded padding to restore the original image size."""
    pad_left, pad_top, pad_right, pad_bottom = padding

    original_width, original_height = image.size
    cropped_image = image.crop((pad_left, pad_top, original_width - pad_right, original_height - pad_bottom))

    return cropped_image


def prep_image_for_return(image):  # take torch image and return PIL
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype("uint8")
    image = Image.fromarray(image)
    return image


def load_im_into_format_from_path(im_path):  # From path get formatted PIL image

    if isinstance(im_path, Image.Image):
        return im_path
    with Image.open(im_path) as image:
        return ImageOps.exif_transpose(image.convert("RGB"))

