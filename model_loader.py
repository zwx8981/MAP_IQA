"""Explicit, reusable model initialization; importing this module loads no weights."""

from dataclasses import dataclass
from typing import Any

from error_context import error_context
from model_config import MODEL_CONFIG, ModelConfig


@dataclass
class QualityEvaluation:
    value: Any
    prediction: float
    mapped_score: float


@dataclass
class EnhancementModels:
    device: str
    dtype: Any
    text_encoder: Any
    tokenizer: Any
    vae: Any
    unet: Any
    controlnet: Any
    ram: Any
    tensor_transform: Any
    ram_transform: Any
    fidelity: Any


def _require_paths(*paths):
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing model paths:\n' + '\n'.join(missing))


def _load_component(name, location, loader):
    with error_context(f'Loading {name} from {location}'):
        return loader()


def _prepare_model(model, device, dtype=None):
    model.to(device, **({'dtype': dtype} if dtype is not None else {}))
    model.eval()
    # Only image latents are optimized; gradients must still flow through the models.
    model.requires_grad_(False)
    return model


def load_models(device=None, *, config: ModelConfig = MODEL_CONFIG):
    """Load shared enhancement models once, preserving the existing weight locations."""
    import torch

    device = config.device if device is None else device
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Image enhancement requires an available CUDA GPU. Activate the iqa environment and check GPU access.')
    _require_paths(
        config.sd_model_path / 'text_encoder', config.sd_model_path / 'tokenizer', config.sd_model_path / 'vae',
        config.sd_model_path / 'unet', config.adapted_model_path / 'unet', config.adapted_model_path / 'controlnet',
        config.ram_weight_path, config.ram_condition_path,
    )
    with error_context('Importing enhancement model implementations'):
        from transformers import CLIPTextModel, CLIPTokenizer
        from diffusers import AutoencoderKL
        from models.controlnet import ControlNetModel
        from models.unet_2d_condition import UNet2DConditionModel
        from ram.models import ram
        from torchvision import transforms
        from nlpd import NLPD

    dtypes = {'float16': torch.float16, 'float32': torch.float32}
    if config.diffusion_weight_dtype not in dtypes:
        raise ValueError(f'Unsupported diffusion weight dtype: {config.diffusion_weight_dtype!r}')
    dtype = dtypes[config.diffusion_weight_dtype]
    text_encoder = _load_component('text encoder', config.sd_model_path / 'text_encoder', lambda: _prepare_model(
        CLIPTextModel.from_pretrained(str(config.sd_model_path), subfolder='text_encoder'), device, dtype,
    ))
    tokenizer = _load_component('tokenizer', config.sd_model_path / 'tokenizer', lambda:
        CLIPTokenizer.from_pretrained(str(config.sd_model_path), subfolder='tokenizer'))
    vae = _load_component('VAE', config.sd_model_path / 'vae', lambda: _prepare_model(
        AutoencoderKL.from_pretrained(str(config.sd_model_path), subfolder='vae'), device, dtype,
    ))
    unet = _load_component('UNet', f'{config.sd_model_path / "unet"} + {config.adapted_model_path / "unet"}', lambda: _prepare_model(
        UNet2DConditionModel.from_pretrained_orig(
            str(config.sd_model_path), str(config.adapted_model_path), subfolder='unet', use_image_cross_attention=config.unet_use_image_cross_attention,
        ), device, dtype,
    ))
    controlnet = _load_component('ControlNet', config.adapted_model_path / 'controlnet', lambda: _prepare_model(
        ControlNetModel.from_pretrained(str(config.adapted_model_path), subfolder='controlnet'), device, dtype,
    ))
    ram_model = _load_component('RAM', f'{config.ram_weight_path} + {config.ram_condition_path}', lambda: _prepare_model(
        ram(pretrained=str(config.ram_weight_path), pretrained_condition=str(config.ram_condition_path), image_size=config.ram_image_size, vit=config.ram_vit),
        device,
    ))
    fidelity = _load_component('NLPD', 'built-in filters', lambda: _prepare_model(
        NLPD(channels=3, k=5, test_y_channel=False, low_freq=True), device,
    ))
    tensor_transform = transforms.Compose([transforms.ToTensor()])
    ram_transform = transforms.Compose([
        transforms.Resize((config.ram_image_size, config.ram_image_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print('Loaded enhancement models')
    return EnhancementModels(
        device=device, dtype=dtype, text_encoder=text_encoder, tokenizer=tokenizer, vae=vae,
        unet=unet, controlnet=controlnet, ram=ram_model, tensor_transform=tensor_transform,
        ram_transform=ram_transform, fidelity=fidelity,
    )


def load_quality_scorer(quality_model='ms_liqe', device=None, grad_scale=1, *, config: ModelConfig = MODEL_CONFIG):
    """Load one selected metric; preserve calibration, half precision and input gradients."""
    from mapping2 import SUPPORTED_QUALITY_MODELS, logistic_mapping

    if quality_model not in SUPPORTED_QUALITY_MODELS:
        raise ValueError(f'Unsupported quality model {quality_model!r}; choose from {SUPPORTED_QUALITY_MODELS}')

    device = config.device if device is None else device

    def load_metric():
        if quality_model == 'ms_liqe':
            _require_paths(config.ms_liqe_weight_path)
            from ms_liqe import MSLIQE
            return MSLIQE(ckpt=str(config.ms_liqe_weight_path), device=device, scale=list(config.ms_liqe_scales))
        import pyiqa
        options = {'num_patch': config.liqe_num_patch} if quality_model == 'liqe' else {}
        if quality_model == 'hyperiqa':
            _require_paths(config.hyperiqa_weight_path)
            options['pretrained_model_path'] = str(config.hyperiqa_weight_path)
        return pyiqa.create_metric(
            'liqe_mix' if quality_model == 'liqe' else quality_model,
            as_loss=True, device=device, **options,
        )

    location = config.ms_liqe_weight_path if quality_model == 'ms_liqe' else (
        config.hyperiqa_weight_path if quality_model == 'hyperiqa' else 'existing PyIQA pretrained weights'
    )
    model = _load_component(f'quality model {quality_model}', location, lambda: _prepare_model(load_metric(), device))
    mono = -1 if quality_model == 'niqe' else 1

    def quality_score(image, model_name):
        if model_name != quality_model:
            raise ValueError(f'Scorer for {quality_model!r} cannot score as {model_name!r}')
        pixels = ((image / 2) + 0.5).clamp(0, 1).float()
        prediction = model(pixels)
        prediction_value = prediction.item()
        print(f'quality prediction:{prediction_value}')
        score = logistic_mapping(mono * prediction, quality_model)
        mapped_value = score.item()
        print(f'mapped prediction:{mapped_value}')
        return QualityEvaluation(score.half() * grad_scale, prediction_value, mapped_value)

    return quality_score
