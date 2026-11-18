"""Run the current MAP_IQA entry point for one existing input image."""

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import stat
import tempfile
import time
import traceback
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "extraction_manifest.json").read_text())
SOURCE = Path(MANIFEST["source"])


class ProtectedFileError(BaseException):
    """Stop before writing outside the extracted project or temporary area."""


def verify_originals():
    for group in ("code_files", "resource_files"):
        for entry in MANIFEST[group]:
            original = SOURCE / entry["path"]
            assert hashlib.sha256(original.read_bytes()).hexdigest() == entry["sha256"], original
    for name, before in MANIFEST["weight_files_before"].items():
        current = Path(name).stat()
        after = {
            "size": current.st_size,
            "mtime_ns": current.st_mtime_ns,
            "inode": current.st_ino,
            "device": current.st_dev,
        }
        assert after == before, f"Weight metadata changed: {name}"


def install_write_guard(scratch):
    allowed = (str(ROOT), str(scratch))
    write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
    written_images = set()

    def resolve(path, dir_fd=-1):
        if isinstance(path, int):
            return os.path.realpath(f"/proc/self/fd/{path}")
        path = os.fsdecode(path)
        if not os.path.isabs(path) and dir_fd not in (-1, None):
            path = os.path.join(os.path.realpath(f"/proc/self/fd/{dir_fd}"), path)
        return os.path.realpath(path)

    def writable(path, dir_fd=-1):
        path = resolve(path, dir_fd)
        if path == str(ROOT / "ms_liqe.pt"):
            return False
        return any(path == root or path.startswith(root + os.sep) for root in allowed)

    def guard(event, args):
        if event == "open":
            path, mode, flags = args
            if isinstance(path, int) and stat.S_ISFIFO(os.fstat(path).st_mode):
                return
            writing = (
                isinstance(mode, str) and any(char in mode for char in "wax+")
            ) or (isinstance(flags, int) and flags & write_flags)
            if writing and path != "/dev/null" and not writable(path):
                raise ProtectedFileError(f"Protected file write: {path!r}")
            if writing and isinstance(path, (str, bytes, os.PathLike)):
                resolved = Path(resolve(path))
                if resolved.suffix.lower() == ".png":
                    written_images.add(resolved)
        elif event == "os.mkdir":
            if not writable(args[0], args[2]) and not os.path.isdir(resolve(args[0], args[2])):
                raise ProtectedFileError(f"Protected directory creation: {args[0]!r}")
        elif event in {"os.remove", "os.rmdir", "os.chmod", "os.chown", "os.truncate", "os.utime"}:
            index = {"os.remove": 1, "os.rmdir": 1, "os.chmod": 2, "os.chown": 3, "os.utime": 3}.get(event)
            dir_fd = args[index] if index is not None else -1
            if not writable(args[0], dir_fd):
                raise ProtectedFileError(f"Protected filesystem operation: {event} {args[0]!r}")
        elif event in {"os.rename", "os.link"}:
            if not (writable(args[0], args[2]) and writable(args[1], args[3])):
                raise ProtectedFileError(f"Protected filesystem operation: {event}")
        elif event == "os.symlink":
            if not writable(args[1], args[2]):
                raise ProtectedFileError(f"Protected symlink creation: {args[1]!r}")
        elif event == "subprocess.Popen":
            command = args[1]
            readonly_commands = (
                ["uname", "-p"], ["/usr/bin/uname", "-p"],
                ["/sbin/ldconfig", "-p"], ["/usr/sbin/ldconfig", "-p"],
                ["/bin/sh", "-c", "pip list | grep habana-torch-plugin"],
            )
            if isinstance(command, (list, tuple)) and (
                os.path.basename(args[0]) == "fc-list" or list(command) in readonly_commands
            ):
                return
            raise ProtectedFileError(f"Protected child process: {args[:2]!r}")
        elif event in {"os.system", "socket.connect", "socket.connect_ex"}:
            raise ProtectedFileError(f"Protected process/network operation: {event}")

    sys.addaudithook(guard)
    return written_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="Canon_041_LR4.png")
    parser.add_argument("--compare-baseline", action="store_true")
    args = parser.parse_args()
    entry = runpy.run_path(str(ROOT / "MAP_IQA.py"), run_name="entry_configuration")
    config = entry["CONFIG"]
    input_dir = entry["resolve_path"](config["input_dir"]).resolve()
    models = config["quality_models"]
    assert Path(args.image).name == args.image, "Use an input filename, not a path"
    image = input_dir / args.image
    assert image.is_file(), image
    if args.compare_baseline and len(models) != 1:
        raise ValueError("Strict baseline comparison requires one model")
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    verify_originals()
    started = time.monotonic()
    report = {
        "status": "running",
        "working_directory": str(ROOT),
        "entrypoint": str(ROOT / "MAP_IQA.py"),
        "input_image": str(image),
        "diffusion_steps": config["diffusion_steps"],
        "maximum_optimization_steps": config["optimization_steps"],
        "quality_model": models[0] if len(models) == 1 else None,
        "quality_models": models,
        "fidelity_weight": config["fidelity_weight"],
        "latent_step_size": config["latent_step_size"],
        "entry_config": config,
        "python": sys.executable,
    }
    result_path = ROOT / "_validation" / "result.json"
    real_listdir = os.listdir
    calls = []

    with tempfile.TemporaryDirectory(prefix="diffusion-extracted-validation-", dir="/tmp") as name:
        scratch = Path(name)
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ["CUDA_CACHE_DISABLE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        os.environ.setdefault("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
        os.environ["TMPDIR"] = name
        os.environ["MPLCONFIGDIR"] = str(scratch / "matplotlib")
        os.environ["KERAS_HOME"] = str(scratch / "keras")
        os.environ["XDG_CACHE_HOME"] = str(scratch / "xdg")
        os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        os.environ["PIP_NO_CACHE_DIR"] = "1"
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(scratch / "torchinductor")
        os.environ["TRITON_CACHE_DIR"] = str(scratch / "triton")
        tempfile.tempdir = name
        written_images = install_write_guard(scratch)

        try:
            import torch
            from PIL import Image

            entry["validate_config"](config)
            report["torch"] = torch.__version__
            report["cuda_available"] = torch.cuda.is_available()
            assert report["cuda_available"], "CUDA must be available for this entry point"
            report["gpu"] = torch.cuda.get_device_name()
            from dataclasses import asdict
            from model_config import MODEL_CONFIG
            report['model_config'] = asdict(MODEL_CONFIG)
            report["packages"] = {
                package: importlib.metadata.version(package)
                for package in ("diffusers", "transformers", "pyiqa", "timm")
            }
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)

            grad_mode_before_import = torch.is_grad_enabled()
            anomaly_mode_before_import = torch.is_anomaly_enabled()
            import MAP_diffusion_latent
            assert torch.is_grad_enabled() == grad_mode_before_import
            assert torch.is_anomaly_enabled() == anomaly_mode_before_import

            real_gen = MAP_diffusion_latent.gen
            expected_kwargs = entry["build_gen_kwargs"](config)
            model_bundles = []

            def checked_gen(*positional, **keywords):
                for key, value in expected_kwargs.items():
                    assert keywords[key] == value, (key, keywords[key], value)
                assert "quality_lambda" not in keywords
                model_name = keywords["quality_model"]
                assert model_name in models
                assert Path(keywords["image_path"]).resolve() == image
                bundle = keywords['models']
                if model_bundles:
                    assert bundle is model_bundles[0], 'Shared models were loaded more than once'
                else:
                    model_bundles.append(bundle)
                for component in ('text_encoder', 'vae', 'unet', 'controlnet', 'ram', 'fidelity'):
                    model = getattr(bundle, component)
                    assert not model.training, component
                    assert all(not parameter.requires_grad for parameter in model.parameters()), component
                assert not keywords['text_embedding'].requires_grad
                grad_mode_before_gen = torch.is_grad_enabled()
                result = real_gen(*positional, **keywords)
                assert torch.is_grad_enabled() == grad_mode_before_gen
                assert all(parameter.grad is None for component in ('text_encoder', 'vae', 'unet', 'controlnet', 'ram', 'fidelity')
                           for parameter in getattr(bundle, component).parameters())
                assert isinstance(result, Image.Image), "gen() did not finish with an image"
                calls.append({
                    "size": list(result.size), "mode": result.mode, "model": model_name,
                    "fidelity_weight": keywords["fidelity_weight"],
                    "latent_step_size": keywords["latent_step_size"], "output_dir": keywords["output_dir"],
                })
                return result

            def one_image_listdir(path="."):
                if isinstance(path, (str, bytes, os.PathLike)) and Path(os.fsdecode(path)).resolve() == input_dir:
                    return [args.image]
                return real_listdir(path)

            MAP_diffusion_latent.gen = checked_gen
            os.listdir = one_image_listdir
            try:
                with patch.object(sys, "argv", [str(ROOT / "MAP_IQA.py")]):
                    runpy.run_path(str(ROOT / "MAP_IQA.py"), run_name="__main__")
            finally:
                os.listdir = real_listdir
                MAP_diffusion_latent.gen = real_gen

            expected_runs = len(models)
            assert len(calls) == expected_runs, (len(calls), expected_runs)
            with Image.open(image) as original:
                from PIL import ImageOps
                expected_size = ImageOps.exif_transpose(original).size
            outputs, intermediates, videos, histories = [], [], [], []

            def describe_image(path):
                assert path.resolve() in written_images, f'Image was not written by this run: {path}'
                with Image.open(path) as result:
                    result.load()
                    assert result.size == expected_size, (path, result.size, expected_size)
                    return {'path': str(path), 'size': list(result.size), 'bytes': path.stat().st_size,
                            'pixel_sha256': hashlib.sha256(result.tobytes()).hexdigest()}

            for call in calls:
                model_dir = Path(call['output_dir']) / call['model']
                output_name = image.stem + '.png'
                outputs.append(describe_image(model_dir / 'final' / output_name))
                history_path = model_dir / 'scores' / (image.stem + '.json')
                history = json.loads(history_path.read_text())
                assert history['frames'][0]['stage'] == 'input'
                assert len(history['frames']) == history['completed_optimization_steps'] + 1
                assert 1 <= history['completed_optimization_steps'] <= config['optimization_steps']
                assert all(row['prediction'] is not None and row['mapped_score'] is not None for row in history['frames'])
                if config['save_intermediates']:
                    for frame in history['frames']:
                        assert frame['image'] is not None
                        intermediates.append(describe_image(model_dir / frame['image']))
                else:
                    assert all(frame['image'] is None for frame in history['frames'])
                histories.append({'path': str(history_path), **history})
                if config['save_video']:
                    import av
                    info = history['video']
                    assert info and info['show_scores'] == config['video_show_scores']
                    video_path = model_dir / info['path']
                    with av.open(str(video_path)) as container:
                        stream = container.streams.video[0]
                        assert stream.codec_context.name == 'h264'
                        assert [stream.width, stream.height] == info['size']
                        frames = sum(1 for _ in container.decode(video=0))
                        assert frames == info['frames']
                    videos.append({'path': str(video_path), 'bytes': video_path.stat().st_size, **info})
                    videos[-1]['path'] = str(video_path)
                else:
                    assert history['video'] is None

            if args.compare_baseline:
                if not config['save_intermediates']:
                    raise ValueError('Strict comparison requires saved intermediate frames')
                baseline = json.loads((ROOT / '_validation/baseline_before_cleanup.json').read_text())
                expected = {int(Path(item['path']).parent.name): item['pixel_sha256'] for item in baseline['outputs']}
                rows = histories[0]['frames'][1:]
                actual = {row['step'] - 1: describe_image(Path(calls[0]['output_dir']) / calls[0]['model'] / row['image'])['pixel_sha256'] for row in rows}
                mismatches = sorted(step for step in expected.keys() | actual.keys() if expected.get(step) != actual.get(step))
                report['baseline_comparison'] = {'pixel_identical': not mismatches, 'compared_images': len(expected), 'mismatches': mismatches}
                assert not mismatches, f'Output differs from baseline: {mismatches}'

            local_modules = {}
            for module_name, module in tuple(sys.modules.items()):
                filename = getattr(module, "__file__", None)
                if not filename:
                    continue
                path = Path(filename).resolve()
                if not path.is_file():
                    continue
                assert SOURCE not in path.parents, f"Imported original project code: {module_name}: {path}"
                if ROOT in path.parents:
                    local_modules[module_name] = str(path)

            unused_ram_modules = {'ram.inference', 'ram.transform', 'ram.models.ram', 'ram.models.tag2text', 'ram.models.bert'}
            assert not unused_ram_modules & set(sys.modules), unused_ram_modules & set(sys.modules)
            application_files = sorted(
                {str(ROOT / "MAP_IQA.py")}
                | {filename for filename in local_modules.values() if ROOT / "_validation" not in Path(filename).parents}
            )
            verify_originals()
            report.update({
                "status": "passed",
                "completed_images": 1,
                "completed_parameter_combinations": len(calls),
                "gen_calls": calls,
                "outputs": outputs,
                "intermediate_outputs": intermediates, "videos": videos, "score_histories": histories,
                "local_modules": local_modules,
                "application_code_files": application_files,
                "application_code_file_count": len(application_files),
                "original_code_and_resources_unchanged": True,
                "weight_metadata_unchanged": True,
                "shared_models_reused": True,
                "unused_RAM_exports_not_imported": True,
                "model_parameters_frozen_and_without_gradients": True,
                "model_eval_mode": True,
                "global_grad_and_anomaly_modes_preserved": True,
            })
        except BaseException as exc:
            os.listdir = real_listdir
            report.update({"status": "failed", "error": str(exc), "error_type": type(exc).__name__})
            traceback.print_exc()
            raise
        finally:
            report["elapsed_seconds"] = round(time.monotonic() - started, 2)
            result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n")
            print("VALIDATION_RESULT", report["status"], str(result_path), flush=True)


if __name__ == "__main__":
    main()
