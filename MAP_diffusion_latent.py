"""Quality-guided image enhancement with coupled EDICT latent optimization."""

from dataclasses import dataclass
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import autocast, nn
from diffusers import DDIMScheduler
from PIL import Image

from helper_functions import (
    forward_step, reverse_step,
    load_im_into_format_from_path, pad_to_multiple_of_64, crop_to_original,
    prep_image_for_return,
)
import memcnn
from error_context import error_context
from mapping2 import SUPPORTED_QUALITY_MODELS
from model_loader import EnhancementModels
from output_manager import EnhancementOutput, OutputOptions, check_video_backend
from tqdm import tqdm

VAE_LATENT_SCALE = 0.18215
MOMENTUM = 0.9
EARLY_STOP_SCORES = {
    'ms_liqe': 96, 'dbcnn': 93, 'liqe': 93.4,
    'hyperiqa': 97, 'musiq': 89, 'clipiqa+': 95,
    'niqe': 85, 'paq2piq': 85, 'unique': 93,
}


class SteppingLayer(nn.Module):
    """One coupled diffusion step; index/time outputs serve the MemCNN interface."""

    def __init__(self, unet, controlnet, text_embedding, scheduler=None, num_timesteps=20):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.text_embedding = text_embedding
        self.num_timesteps = num_timesteps
        if scheduler is None:
            scheduler = DDIMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                num_train_timesteps=1000, clip_sample=False, set_alpha_to_one=False,
            )
        self.scheduler = scheduler
        self.scheduler.set_timesteps(num_timesteps)

    def forward(self, i, t, latent_pair, ram_features, control_image, reverse=False):
        for base_latent_i in range(2):
            if reverse:
                original_index = self.num_timesteps - (i + 1)
                offset = (original_index + 1) % 2
            else:
                offset = i % 2
            latent_i = ((base_latent_i + offset) % 2).long()
            latent_j = ((latent_i + 1) % 2).long()

            model_input = latent_pair.index_select(0, latent_j) if base_latent_i == 0 else first_output
            latent_base = latent_pair.index_select(0, latent_i)
            down_residuals, mid_residual = self.controlnet(
                model_input, t, encoder_hidden_states=self.text_embedding,
                controlnet_cond=control_image, conditioning_scale=1,
                guess_mode=False, return_dict=False,
                image_encoder_hidden_states=ram_features,
            )
            noise_prediction = self.unet(
                model_input, t[0], encoder_hidden_states=self.text_embedding,
                cross_attention_kwargs=None, down_block_additional_residuals=down_residuals,
                mid_block_additional_residual=mid_residual, return_dict=False,
                image_encoder_hidden_states=ram_features,
            )[0]
            step = reverse_step if reverse else forward_step
            updated = step(self.scheduler, noise_prediction, t[0].long(), latent_base)
            updated = updated.to(latent_base.dtype)
            if base_latent_i == 0:
                first_output = updated
            elif latent_i == 1:
                combined = torch.cat([first_output, updated])
            else:
                combined = torch.cat([updated, first_output])
        return i.clone(), t.clone(), combined

    def inverse(self, i, t, latent_pair, ram_features, control_image):
        return self.forward(i, t, latent_pair, ram_features, control_image, reverse=True)


class MixingLayer(nn.Module):
    """
    This does the mixing layer of EDICT
    https://arxiv.org/abs/2211.12446
    Equations 12/13
    """

    def __init__(self, mix_weight=0.93):
        super(MixingLayer, self).__init__()
        self.p = mix_weight

    def forward(self, input_x):
        input_x0, input_x1 = input_x[:1], input_x[1:]
        x0 = self.p * input_x0 + (1 - self.p) * input_x1
        x1 = (1 - self.p) * x0 + self.p * input_x1
        return torch.cat([x0, x1])

    def inverse(self, input_x):
        input_x0, input_x1 = input_x.split(1)
        x1 = (input_x1 - (1 - self.p) * input_x0) / self.p
        x0 = (input_x0 - (1 - self.p) * x1) / self.p
        return torch.cat([x0, x1])


@dataclass
class PreparedImage:
    source_tensor: torch.Tensor
    control_tensor: torch.Tensor
    ram_features: torch.Tensor
    padding: tuple
    original: Image.Image


def _prepare_image(image_path, models):
    image = load_im_into_format_from_path(str(image_path))
    original = image
    image, padding = pad_to_multiple_of_64(image)

    ram_input = models.tensor_transform(image).unsqueeze(0).to(models.device)
    ram_features = models.ram.generate_image_embeds(models.ram_transform(ram_input))
    pixels = np.array(image) / 255.0 * 2.0 - 1.0
    source = torch.from_numpy(pixels[np.newaxis, ...].transpose(0, 3, 1, 2))
    source = source.to(models.device).to(models.unet.dtype)
    return PreparedImage(source, source * 0.5 + 0.5, ram_features, padding, original)


def _build_diffusion_process(models, text_embedding, diffusion_steps, mix_weight):
    mixing = memcnn.InvertibleModuleWrapper(
        MixingLayer(mix_weight), keep_input=True, keep_input_inverse=True, num_bwd_passes=1,
    )
    stepping = SteppingLayer(
        models.unet, models.controlnet, text_embedding, num_timesteps=diffusion_steps,
    )
    stepping = memcnn.InvertibleModuleWrapper(
        stepping, keep_input=True, keep_input_inverse=True, num_bwd_passes=1,
    )
    return stepping, mixing, stepping._fn.scheduler.timesteps


def _run_diffusion(latents, timesteps, stepping, mixing, image, inverse=False):
    for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
        i = torch.tensor([i], dtype=latents.dtype, device=latents.device)
        t = torch.tensor([t], dtype=latents.dtype, device=latents.device)
        if inverse:
            latents = mixing.inverse(latents)
            i, t, latents = stepping.inverse(i, t, latents, image.ram_features, image.control_tensor)
        else:
            i, t, latents = stepping(i, t, latents, image.ram_features, image.control_tensor)
            latents = mixing(latents)
    return latents


def _initialize_latents(image, stepping, mixing, timesteps, generator, models):
    with autocast(models.device):
        initial = models.vae.encode(image.source_tensor).latent_dist.sample(generator=generator) * VAE_LATENT_SCALE
        latents = initial.repeat(2, 1, 1, 1)
        return _run_diffusion(latents, timesteps.flip(0), stepping, mixing, image, inverse=True)


def _decode_latents(latents, vae):
    return [vae.decode(part.to(vae.dtype) / VAE_LATENT_SCALE).sample for part in latents.chunk(2)]


def _compute_objective(candidate, source, quality_score_fn, quality_model,
                       fidelity_fn, *, fidelity_weight=100, image_path, iteration, candidate_index):
    with error_context(
        f"Quality model {quality_model!r} failed for {image_path}, "
        f"optimization step {iteration}, candidate {candidate_index}"
    ):
        evaluation = quality_score_fn(candidate, quality_model)
        quality_score = evaluation.value
        if not all(math.isfinite(value) for value in (quality_score.item(), evaluation.prediction, evaluation.mapped_score)):
            raise FloatingPointError('Quality score is not finite')
    objective = quality_score - fidelity_weight * fidelity_fn(
        (0.5 * candidate + 0.5).clamp(0, 1).float(),
        (0.5 * source + 0.5).clamp(0, 1).float(),
    ).half()
    if not torch.isfinite(objective).all():
        raise FloatingPointError(
            f'Non-finite objective for {image_path}, optimization step {iteration}, candidate {candidate_index}'
        )
    return objective.half(), evaluation


def _update_latents(trainable_latents, previous_momenta, iteration, clip_grad_val,
                    use_momentum, use_nesterov, tied_latents, *, latent_step_size=2):
    if trainable_latents.grad is None:
        raise RuntimeError('No latent gradient was produced by the objective')
    if not torch.isfinite(trainable_latents.grad).all():
        raise FloatingPointError('Latent gradient is not finite')
    gradient = 0.5 * trainable_latents.grad
    if tied_latents:
        gradient = gradient.mean(dim=0, keepdim=True)
        gradient = gradient.repeat(2, 1, 1, 1)
    updated_latents = []
    for index, (grad, latent) in enumerate(zip(gradient.chunk(2), trainable_latents.chunk(2))):
        if clip_grad_val is not None:
            grad = grad.clip(-clip_grad_val, clip_grad_val)
        if use_momentum:
            momentum = grad if iteration == 0 else MOMENTUM * previous_momenta[index] + grad
            grad = grad + MOMENTUM * momentum if use_nesterov else momentum
            previous_momenta[index] = momentum.clone()
        # The objective is maximized, so this is gradient ascent.
        updated_latents.append((latent + latent_step_size * grad).clone())
    if tied_latents:
        mean_latent = 0.5 * (updated_latents[0] + updated_latents[1])
        return mean_latent.repeat(2, 1, 1, 1)
    return torch.cat(updated_latents)


def _optimize_latents(latents, image, stepping, mixing, timesteps, *, models, quality_score_fn,
                      quality_model, fidelity_weight, fidelity_fn, image_path, outputs,
                      optimization_steps, latent_step_size, clip_grad_val, use_momentum,
                      use_nesterov, tied_latents):
    previous_momenta = [None, None]
    early_stop = False
    with torch.enable_grad():
        for iteration in range(optimization_steps):
            if early_stop:
                print('Early stopping!')
                break
            print(f"Optimization Step {iteration}")
            with error_context(f'Optimization step {iteration}'):
                trainable_latents = latents.clone().detach().requires_grad_(True)
                with autocast(models.device):
                    generated_latents = _run_diffusion(
                        trainable_latents.clone(), timesteps, stepping, mixing, image,
                    )
                    candidates = _decode_latents(generated_latents, models.vae)
                objectives = []
                for candidate_index, candidate in enumerate(candidates):
                    objective, evaluation = _compute_objective(
                        candidate, image.source_tensor, quality_score_fn, quality_model,
                        fidelity_fn, image_path=image_path,
                        iteration=iteration, candidate_index=candidate_index, fidelity_weight=fidelity_weight,
                    )
                    objectives.append(objective)
                    if candidate_index == 0:
                        early_stop = evaluation.value.item() >= EARLY_STOP_SCORES[quality_model]
                        result = crop_to_original(prep_image_for_return(candidate.detach()), image.padding)
                        outputs.add_frame(result, evaluation, step=iteration + 1)
                sum(objectives).backward()
                latents = _update_latents(
                    trainable_latents, previous_momenta, iteration, clip_grad_val,
                    use_momentum, use_nesterov, tied_latents,
                    latent_step_size=latent_step_size,
                )
    return result


def _validate_generation(image_path, quality_model, quality_score_fn, text_embedding,
                         diffusion_steps, optimization_steps, fidelity_weight,
                         mix_weight, clip_grad_val, latent_step_size):
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")
    if quality_model not in SUPPORTED_QUALITY_MODELS:
        raise ValueError(f"Unsupported quality model: {quality_model!r}")
    if not callable(quality_score_fn) or text_embedding is None:
        raise ValueError('A quality score function and text embedding are required')
    if type(diffusion_steps) is not int or not 1 <= diffusion_steps <= 1000:
        raise ValueError('diffusion_steps must be an integer from 1 to 1000')
    if type(optimization_steps) is not int or optimization_steps < 1:
        raise ValueError('optimization_steps must be a positive integer')
    if type(latent_step_size) not in (int, float) or not math.isfinite(latent_step_size) or latent_step_size <= 0:
        raise ValueError('latent_step_size must be finite and positive')
    if type(fidelity_weight) not in (int, float) or not math.isfinite(fidelity_weight) or fidelity_weight < 0:
        raise ValueError('fidelity_weight must be finite and nonnegative')
    if not math.isfinite(mix_weight) or not 0 < mix_weight <= 1:
        raise ValueError('mix_weight must be finite and in (0, 1]')
    if clip_grad_val is not None and (not math.isfinite(clip_grad_val) or clip_grad_val <= 0):
        raise ValueError('clip_grad_val must be positive or None')


def _evaluate_input(source, quality_score_fn, quality_model):
    """Record input IQA without consuming randomness used by optimization."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [source.device] if source.is_cuda else []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad(), error_context('Scoring input image'):
            evaluation = quality_score_fn(source, quality_model)
            if not all(math.isfinite(value) for value in (
                evaluation.value.item(), evaluation.prediction, evaluation.mapped_score,
            )):
                raise FloatingPointError('Input IQA score is not finite')
            return evaluation
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def gen(*, models: EnhancementModels, image_path, quality_model, quality_score_fn, text_embedding, output_dir,
        output_name=None, diffusion_steps=20, optimization_steps=15, fidelity_weight=100,
        mix_weight=0.93, latent_step_size=2, seed=0, clip_grad_val=1e-3, use_momentum=True,
        use_nesterov=False, tied_latents=False, output_options: OutputOptions = OutputOptions()) -> Image.Image:
    """Enhance one existing image and return the last saved PIL image.

    Reuse the explicit model bundle returned by model_loader.load_models().
    Output options control intermediate PNGs and progression videos. The final
    image and score history are saved separately for every successful image.
    """
    image_path = Path(image_path)
    _validate_generation(
        image_path, quality_model, quality_score_fn, text_embedding, diffusion_steps,
        optimization_steps, fidelity_weight, mix_weight, clip_grad_val, latent_step_size,
    )
    output_options.validate()
    if output_options.save_video:
        check_video_backend()
    output_name = Path(output_name or image_path.name).stem + '.png'
    use_momentum = use_momentum or use_nesterov
    generator = torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    context = f'{image_path} (quality model={quality_model}, fidelity_weight={fidelity_weight})'
    with error_context(f'Preparing diffusion for {context}'):
        stepping, mixing, timesteps = _build_diffusion_process(models, text_embedding, diffusion_steps, mix_weight)
    with torch.no_grad():
        with error_context(f'Preparing image and RAM features for {context}'):
            image = _prepare_image(image_path, models)
        with error_context(f'Initializing latents for {context}'):
            latents = _initialize_latents(image, stepping, mixing, timesteps, generator, models)
    with error_context(f'Enhancing {context}'), EnhancementOutput(
        output_dir, quality_model, output_name, image_path, output_options,
    ) as outputs:
        initial_evaluation = _evaluate_input(image.source_tensor, quality_score_fn, quality_model)
        outputs.add_frame(image.original, initial_evaluation, step=0)
        result = _optimize_latents(
            latents, image, stepping, mixing, timesteps, models=models,
            quality_score_fn=quality_score_fn, quality_model=quality_model,
            fidelity_weight=fidelity_weight, fidelity_fn=models.fidelity, image_path=image_path,
            outputs=outputs, optimization_steps=optimization_steps, clip_grad_val=clip_grad_val,
            latent_step_size=latent_step_size,
            use_momentum=use_momentum, use_nesterov=use_nesterov, tied_latents=tied_latents,
        )
        outputs.finish(result)
        return result
