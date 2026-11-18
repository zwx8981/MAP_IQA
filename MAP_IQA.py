"""Enhance a folder of images. Command-line arguments override CONFIG defaults."""

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path


# ==================== Default run settings ====================
CONFIG = {
    # Relative paths are resolved from this file's directory.
    "input_dir": "initial_images",
    "output_dir": "fidelity_perception_tradeoff/fidelity_100_steps_25",

    # Choose one or more IQA models.
    # ms_liqe, dbcnn, liqe, hyperiqa, musiq, clipiqa+, niqe, paq2piq, unique
    "quality_models": ["ms_liqe"],
    "fidelity_weight": 100,         # Set to 0 to disable the fidelity penalty.

    # Diffusion and latent optimization.
    "diffusion_steps": 20,          # Steps per diffusion pass (1–1000).
    "optimization_steps": 25,       # Maximum iterations, with early stopping at the quality threshold.
    "latent_step_size": 2,          # Gradient ascent step size, applied after clipping and momentum.
    "mix_weight_base": 0.93,        # Effective mix_weight = mix_weight_base ** (50 / diffusion_steps).
    "grad_scale": 1,                # Scales the IQA score used for optimization and early stopping.
    "clip_grad_val": 1e-3,          # Set to None to disable gradient clipping.

    # Final images and scores are always saved. Intermediate PNGs and videos have separate switches.
    "save_intermediates": True,
    "save_video": True,              # Starts at the EDICT reconstruction and fits all steps into 7 seconds.
    "video_show_scores": True,       # Show raw IQA predictions, calibrated scores, and a score curve.

    # Random seed and optimizer options.
    "seed": 0,                      # Reuse the seed for each image. GPU runs may still differ at the pixel level.
    "use_momentum": True,
    "use_nesterov": False,          # Enabling Nesterov also enables momentum.
    "tied_latents": False,          # Average the two latents after each update.

    # Optional text prompt for diffusion conditioning.
    "text_prompt": "",
}
# ==========================================================

PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = (".jpg", ".png")


def _optional_float(value):
    return None if value.lower() == "none" else float(value)


def parse_config(argv=None):
    """Read command-line overrides without changing the defaults in CONFIG."""
    parser = argparse.ArgumentParser(
        description="Enhance a folder of images with diffusion-latent MAP optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.set_defaults(**deepcopy(CONFIG))

    def add(name, **options):
        flags = ["--" + name.replace("_", "-")]
        if "_" in name:
            flags.append("--" + name)
        parser.add_argument(*flags, dest=name, **options)

    add("input_dir", help="Folder containing JPG/PNG images")
    add("output_dir", help="Folder for enhanced images, scores and optional videos")
    add("quality_models", nargs="+", metavar="MODEL", help="One or more IQA models; see README for supported names")
    add("fidelity_weight", type=float, help="NLPD fidelity penalty; 0 disables it")

    add("diffusion_steps", type=int, help="Steps per diffusion pass (1-1000)")
    add("optimization_steps", type=int, help="Maximum optimization iterations, with score-based early stopping")
    add("latent_step_size", type=float, help="Latent gradient ascent step size")
    add("mix_weight_base", type=float, help="Base for mix_weight = base ** (50 / diffusion_steps)")
    add("grad_scale", type=float, help="Quality score scaling, including the early-stop score")
    add("clip_grad_val", type=_optional_float, metavar="FLOAT|none", help="Gradient clipping bound; none disables clipping")

    add("save_intermediates", action=argparse.BooleanOptionalAction, help="Save the input and each iteration's image")
    add("save_video", action=argparse.BooleanOptionalAction, help="Export a 7-second video starting at Step 1")
    add("video_show_scores", action=argparse.BooleanOptionalAction, help="Show IQA values and a score curve in videos")

    add("seed", type=int, help="Random seed used for each image")
    add("use_momentum", action=argparse.BooleanOptionalAction, help="Enable momentum")
    add("use_nesterov", action=argparse.BooleanOptionalAction, help="Enable Nesterov momentum (also enables momentum)")
    add("tied_latents", action=argparse.BooleanOptionalAction, help="Average the two latents after each update")
    add("text_prompt", help="Diffusion text prompt; an empty string is allowed")

    config = vars(parser.parse_args(argv))
    try:
        validate_config(config)
    except ValueError as error:
        parser.error(str(error))
    return config


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def validate_config(config):
    from mapping2 import SUPPORTED_QUALITY_MODELS

    models = config["quality_models"]
    if not isinstance(models, (list, tuple)) or not models:
        raise ValueError("quality_models 必须是非空列表或元组。")
    if any(model not in SUPPORTED_QUALITY_MODELS for model in models):
        raise ValueError(f"quality_models 只支持：{SUPPORTED_QUALITY_MODELS}")
    if len(set(models)) != len(models):
        raise ValueError("quality_models 中不能重复设置同一模型。")

    value = config["fidelity_weight"]
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("fidelity_weight 必须是有限、非负的数值。")

    for key in ("diffusion_steps", "optimization_steps"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} 必须是正整数。")
    if config["diffusion_steps"] > 1000:
        raise ValueError("diffusion_steps 不能超过调度器的 1000 个训练时间步。")
    for key in ("mix_weight_base", "latent_step_size", "grad_scale", "clip_grad_val"):
        value = config[key]
        if key == "clip_grad_val" and value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} 必须是有限的正数。")
    if config["mix_weight_base"] > 1 or build_gen_kwargs(config)["mix_weight"] == 0:
        raise ValueError("mix_weight_base 必须在 (0, 1] 内，计算后的混合权重不能为 0。")
    build_gen_kwargs(config)["output_options"].validate()
    if type(config["seed"]) is not int:
        raise ValueError("seed 必须是整数。")
    for key in ("use_momentum", "use_nesterov", "tied_latents"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} 必须是 True 或 False。")
    if not isinstance(config["text_prompt"], str):
        raise ValueError("text_prompt 必须是字符串。")
    if not resolve_path(config["input_dir"]).is_dir():
        raise ValueError(f"输入目录不存在：{resolve_path(config['input_dir'])}")


def build_gen_kwargs(config):
    from output_manager import OutputOptions

    return {
        "output_options": OutputOptions(**{key: config[key] for key in (
            "save_intermediates", "save_video", "video_show_scores",
        )}),
        "tied_latents": config["tied_latents"],
        "fidelity_weight": config["fidelity_weight"],
        "latent_step_size": config["latent_step_size"],
        "optimization_steps": config["optimization_steps"],
        "diffusion_steps": config["diffusion_steps"],
        "mix_weight": config["mix_weight_base"] ** (50 / config["diffusion_steps"]),
        "clip_grad_val": config["clip_grad_val"],
        "seed": config["seed"],
        "use_momentum": config["use_momentum"],
        "use_nesterov": config["use_nesterov"],
    }


def build_text_embedding(text_prompt, tokenizer, encoder, device, dtype):
    from torch import autocast, no_grad

    with no_grad(), autocast(device):
        tokens = tokenizer(
            text_prompt, padding="max_length", max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt", return_overflowing_tokens=True,
        )
        return encoder(tokens.input_ids.to(device)).last_hidden_state.to(dtype)


def main(config=None):
    config = CONFIG if config is None else config
    validate_config(config)
    input_dir = resolve_path(config["input_dir"])
    output_dir = resolve_path(config["output_dir"])
    images = sorted([
        name for name in os.listdir(input_dir)
        if name.lower().endswith(IMAGE_EXTENSIONS) and (input_dir / name).is_file()
    ], key=str.casefold)
    if not images:
        raise ValueError(f"输入目录中没有 JPG 或 PNG 图片：{input_dir}")

    stems = [Path(name).stem.casefold() for name in images]
    if len(set(stems)) != len(stems):
        raise ValueError('输入目录存在同名但扩展名不同的图片，请先重命名，避免输出覆盖。')
    gen_kwargs = build_gen_kwargs(config)
    print(json.dumps({"config": config, "effective_mix_weight": gen_kwargs["mix_weight"],
                      "image_count": len(images)}, ensure_ascii=False, indent=2, default=str))

    previous_dir = Path.cwd()
    try:
        # Outputs are resolved from the project directory. The model loader resolves checkpoint paths separately.
        os.chdir(PROJECT_DIR)
        from model_config import MODEL_CONFIG
        from model_loader import load_models, load_quality_scorer
        from MAP_diffusion_latent import gen
        from error_context import error_context
        from output_manager import check_video_backend
        from tqdm import tqdm

        output_dir.mkdir(parents=True, exist_ok=True)
        if config["save_video"]:
            check_video_backend()
        models = load_models(config=MODEL_CONFIG)
        with error_context('Encoding the diffusion text prompt'):
            text_embedding = build_text_embedding(
                config["text_prompt"], models.tokenizer, models.text_encoder, models.device, models.dtype,
            )

        for model in config["quality_models"]:
            quality_score_fn = load_quality_scorer(model, device=models.device, grad_scale=config["grad_scale"], config=MODEL_CONFIG)
            for name in tqdm(images, desc=model):
                gen(
                    **gen_kwargs, models=models, quality_model=model,
                    output_name=Path(name).stem + ".png", image_path=str(input_dir / name),
                    quality_score_fn=quality_score_fn, text_embedding=text_embedding,
                    output_dir=str(output_dir),
                )
    finally:
        os.chdir(previous_dir)


if __name__ == "__main__":
    main(parse_config())
