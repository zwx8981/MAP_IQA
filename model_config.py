"""Model locations and loading options; edit defaults here without importing Torch."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


PROJECT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelConfig:
    # Existing files and links stay in place; these are absolute references only.
    sd_model_path: Path = PROJECT_DIR.parent / 'stable-diffusion-2-base'
    adapted_model_path: Path = PROJECT_DIR / 'preset/models/seesr'
    ram_weight_path: Path = PROJECT_DIR / 'preset/models/ram_swin_large_14m.pth'
    ram_condition_path: Path = PROJECT_DIR / 'preset/DAPE.pth'
    ms_liqe_weight_path: Path = PROJECT_DIR / 'ms_liqe.pt'
    hyperiqa_weight_path: Path = Path('/home/ps/.cache/torch/hub/checkpoints/HyperIQA-resnet50-koniq10k-48579ec9.pth')

    device: str = 'cuda'
    diffusion_weight_dtype: str = 'float16'
    unet_use_image_cross_attention: bool = False
    ram_image_size: int = 384
    ram_vit: str = 'swin_l'
    ms_liqe_scales: Tuple[float, ...] = (1, 0.75)
    liqe_num_patch: int = 15


MODEL_CONFIG = ModelConfig()
