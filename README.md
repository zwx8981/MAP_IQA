# When No-Reference Image Quality Models Meet MAP Estimation in Diffusion Latents

Implementation of the paper accepted by **IEEE Transactions on Image Processing (TIP)**.
[Paper](https://arxiv.org/abs/2403.06406)

We enhance images by optimizing diffusion latents with a no-reference image quality model and an NLPD fidelity term. This implementation uses ControlNet-augmented Diffusion and can save the intermediate images and a video of the optimization process.

## Demo

Image enhancement progression in a 3×3 grid.

https://github.com/user-attachments/assets/67ffed10-725f-4a6b-83cf-84cdf638e975

## Requirements

A CUDA-capable NVIDIA GPU is required. The tested environment is Linux with Python **3.9.18** and the following main dependencies:

| Package | Tested version |
| --- | --- |
| PyTorch / torchvision | `2.8.0` / `0.23.0` (CUDA build) |
| diffusers | `0.33.1` |
| transformers | `4.50.0` |
| pyiqa | `0.1.13` |
| timm | `1.0.24` |
| loralib / fairscale | `0.1.2` / `0.4.13` |
| accelerate / safetensors | `1.10.1` / `0.5.3` |
| NumPy / SciPy | `1.26.4` / `1.13.1` |
| Pillow / tqdm | `10.4.0` / `4.67.2` |
| PyAV (`av`, for video export) | `14.4.0` |

The default `ms_liqe` model also requires [OpenAI CLIP](https://github.com/openai/CLIP), imported as `clip`. Video export uses PyAV with the **libx264** encoder. Use `--no-save-video` if you do not need videos.

## Model weights

Place the files as shown below to use the defaults in [model_config.py](model_config.py). Paths are relative to the folder containing `MAP_IQA.py`.

| What to download | Where to put it |
| --- | --- |
| [Stable Diffusion 2-base](https://huggingface.co/sd2-community/stable-diffusion-2-base/tree/main). Keep the `text_encoder/`, `vae/`, `unet/`, and `tokenizer/` subfolders. | `../stable-diffusion-2-base/`, as a sibling of this project directory |
| SeeSR [UNet](https://huggingface.co/CSWRY/SeeSR/tree/main/seesr/unet) and [ControlNet](https://huggingface.co/CSWRY/SeeSR/tree/main/seesr/controlnet). Each folder needs its `config.json` and `diffusion_pytorch_model.safetensors`. | `preset/models/seesr/unet/` and `preset/models/seesr/controlnet/` |
| [RAM Swin-L checkpoint](https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth) | `preset/models/ram_swin_large_14m.pth` |
| [DAPE checkpoint](https://huggingface.co/CSWRY/SeeSR/blob/main/DAPE.pth) | `preset/DAPE.pth` |
| This project's fine-tuned MS-LIQE inference checkpoint (354 MB) | `ms_liqe.pt`, next to `MAP_IQA.py` |
| [OpenAI CLIP ViT-B/32](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) | `~/.cache/clip/ViT-B-32.pt`, downloaded automatically on first use if missing |

For SD2-base, keep `config.json` in each model subfolder and the full `tokenizer/` directory. The weight filenames are `text_encoder/model.safetensors`, `unet/diffusion_pytorch_model.safetensors`, and `vae/diffusion_pytorch_model.safetensors`.

RAM also caches the [BERT tokenizer](https://huggingface.co/google-bert/bert-base-uncased) automatically. For offline runs, populate the BERT tokenizer and OpenAI CLIP caches before starting. The other IQA options use [PyIQA checkpoints](https://huggingface.co/chaofengc/IQA-PyTorch-Weights/tree/main). If you select `hyperiqa`, set `hyperiqa_weight_path` in `model_config.py` to the local `HyperIQA-resnet50-koniq10k-48579ec9.pth` file.

## Run

With the weights in place, put your JPG or PNG images in `initial_images/` and run from the project directory.

```bash
python MAP_IQA.py
```

The default run uses `ms_liqe` and writes to `fidelity_perception_tradeoff/fidelity_100_steps_25/`. Images are padded to multiples of 64 during processing and cropped back to their original size before saving. Only files directly inside the input folder are processed.

Command-line arguments override `CONFIG` in [MAP_IQA.py](MAP_IQA.py) for the current run.

```bash
python MAP_IQA.py --input-dir initial_images --output-dir results \
  --quality-models ms_liqe --fidelity-weight 100 --optimization-steps 25
```

Use `--no-save-intermediates` to skip intermediate PNGs, `--no-save-video` to skip videos, or `--no-video-show-scores` to hide the video score panel. `--quality-models` accepts multiple space-separated names, and `--clip-grad-val none` disables gradient clipping. Underscore aliases such as `--fidelity_weight` are also supported. Run `python MAP_IQA.py --help` for the full argument list.

## Configuration

The objective is `quality_score - fidelity_weight * NLPD`. Here, `quality_score` is the calibrated IQA prediction scaled by `grad_scale`.

| Parameter | Default | Description |
| --- | --- | --- |
| `quality_models` | `["ms_liqe"]` | One or more IQA models |
| `fidelity_weight` | `100` | Fidelity penalty; `0` removes this term |
| `latent_step_size` | `2` | Latent gradient ascent step size |
| `diffusion_steps` | `20` | Steps per diffusion pass |
| `optimization_steps` | `25` | Maximum iterations, with score-based early stopping |
| `clip_grad_val` | `1e-3` | Gradient clipping bound; `None` disables clipping |
| `save_intermediates` | `True` | Save the input and each iteration's result |
| `save_video` | `True` | Export a progression video |
| `video_show_scores` | `True` | Show IQA values and a score curve in the video |

Supported IQA models:

```text
ms_liqe, dbcnn, liqe, hyperiqa, musiq, clipiqa+, niqe, paq2piq, unique
```

See `CONFIG` for the remaining optimization settings. Device, weight precision, and checkpoint paths are configured in `model_config.py`.

## Outputs

Results are grouped by IQA model:

```text
output_dir/
└── ms_liqe/
    ├── final/image.png
    ├── intermediate/image/
    │   ├── 000_input.png
    │   ├── 001_step.png
    │   └── ...
    ├── videos/image.mp4
    └── scores/image.json
```

Final images and JSON score histories are always saved. Intermediate PNGs and videos have separate switches. When video export is enabled without intermediate PNGs, temporary frames are removed after encoding.

Videos start at `Step 1`, the initial EDICT reconstruction before the first gradient update. Each video lasts 7 seconds, with one frame per completed step, including when optimization stops early. The optional side panel shows the IQA score curve from `Step 1` and the score change from the original input. The original input is recorded as `Step 0` in the score history and saved as `000_input.png` when intermediate PNGs are enabled.

## Validation

To process one image from the configured input folder:

```bash
python _validation/verify_one.py --image 04655.png
```

The default `ms_liqe` setup has been tested through complete enhancement runs and video encoding/decoding. The other IQA options have been checked for model selection, score calibration, and gradient flow.

## Citation

```bibtex
@article{zhang2024nr_iqa_map,
  title={When No-Reference Image Quality Models Meet MAP Estimation in Diffusion Latents},
  author={Weixia Zhang and Dingquan Li and Guangtao Zhai and Xiaokang Yang and Kede Ma},
  year={2026},
  journal={IEEE Transactions on Image Processing}
}
```

## Acknowledgements

This project builds on the code and pretrained models from [SeeSR](https://github.com/cswry/SeeSR). We thank the authors for making their work available.
