"""Run the configured image folder once, recording progress and verified outputs."""

import sys
sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import json
import os
import re
from pathlib import Path
import tempfile
import time
import traceback

from verify_one import install_write_guard, verify_originals

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run-name', default='initial_images', help='Prefix for validation reports and score CSV')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_name):
        parser.error('--run-name may contain only letters, digits, underscores and hyphens')
    report_path = ROOT / '_validation' / (args.run_name + '_run.json')
    csv_path = ROOT / '_validation' / (args.run_name + '_scores.csv')
    import MAP_IQA as entry
    config = entry.CONFIG.copy()
    input_dir, output_dir = entry.resolve_path(config['input_dir']), entry.resolve_path(config['output_dir'])
    inputs = sorted((path for path in input_dir.iterdir() if path.is_file()
                     and path.suffix.lower() in entry.IMAGE_EXTENSIONS), key=lambda path: path.name.casefold())
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    critical_files = ('MAP_IQA.py', 'MAP_diffusion_latent.py', 'model_loader.py', 'model_config.py',
                      'helper_functions.py', 'output_manager.py', 'ms_liqe.py')
    code_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in critical_files}
    fingerprint = hashlib.sha256(json.dumps([config, hashes, code_hashes], sort_keys=True).encode()).hexdigest()
    weight_stat = (ROOT / 'ms_liqe.pt').stat()
    weight_before = [weight_stat.st_size, weight_stat.st_mtime_ns, weight_stat.st_ino]
    results = []
    if args.resume and report_path.exists():
        previous = json.loads(report_path.read_text())
        assert previous['fingerprint'] == fingerprint, 'Configuration, code or inputs changed; start a fresh run'
        results = [row for row in previous['results'] if row['status'] == 'passed']
    completed = {(row['model'], row['image']) for row in results}
    expected = len(inputs) * len(config['quality_models'])
    report = {'status': 'running', 'run_name': args.run_name, 'input_dir': str(input_dir), 'output_dir': str(output_dir),
              'config': config, 'fingerprint': fingerprint, 'input_sha256': hashes,
              'application_sha256': code_hashes, 'expected_runs': expected, 'results': results,
              'completed_runs': len(completed), 'current': None}
    started = time.monotonic()

    def save():
        report['elapsed_seconds_this_process'] = round(time.monotonic() - started, 2)
        temporary = report_path.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(report_path)
        columns = ('image', 'model', 'status', 'iterations', 'input_iqa', 'final_iqa', 'iqa_change',
                   'input_mapped', 'final_mapped', 'elapsed_seconds', 'final_image', 'video', 'error')
        with csv_path.open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(report['results'])

    verify_originals()
    with tempfile.TemporaryDirectory(prefix='diffusion-folder-run-', dir='/tmp') as name:
        scratch = Path(name)
        os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        os.environ.setdefault('TORCH_HOME', os.path.expanduser('~/.cache/torch'))
        os.environ.update({
            'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_CACHE_DISABLE': '1',
            'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_TELEMETRY': '1',
            'TMPDIR': name, 'MPLCONFIGDIR': str(scratch / 'matplotlib'), 'KERAS_HOME': str(scratch / 'keras'),
            'XDG_CACHE_HOME': str(scratch / 'xdg'), 'TORCHINDUCTOR_COMPILE_THREADS': '1',
            'TORCHINDUCTOR_CACHE_DIR': str(scratch / 'inductor'), 'TRITON_CACHE_DIR': str(scratch / 'triton'),
        })
        tempfile.tempdir = name
        written_images = install_write_guard(scratch)
        import torch
        import av
        from PIL import Image, ImageOps
        entry.validate_config(config)
        assert torch.cuda.is_available(), 'CUDA is required'
        report['gpu'] = torch.cuda.get_device_name()
        import MAP_diffusion_latent as diffusion
        real_gen, real_update = diffusion.gen, diffusion._update_latents
        bundles = []

        def tracked_update(*positional, **keywords):
            value = real_update(*positional, **keywords)
            report['current']['completed_steps'] = int(positional[2]) + 1
            save()
            return value

        def checked_gen(*positional, **keywords):
            image_path = Path(keywords['image_path'])
            model = keywords['quality_model']
            if (model, image_path.name) in completed:
                print('BATCH_SKIP', model, image_path.name, flush=True)
                return None
            report['current'] = {'image': image_path.name, 'model': model, 'completed_steps': 0}
            save()
            image_started = time.monotonic()
            print('BATCH_IMAGE_START', model, image_path.name, flush=True)
            row = {'image': image_path.name, 'model': model}
            try:
                bundle = keywords['models']
                if bundles:
                    assert bundle is bundles[0], 'Shared models were reloaded'
                else:
                    bundles.append(bundle)
                result = real_gen(*positional, **keywords)
                assert isinstance(result, Image.Image)
                with Image.open(image_path) as source:
                    size = ImageOps.exif_transpose(source).size
                assert result.size == size
                model_dir = Path(keywords['output_dir']) / model
                final = model_dir / 'final' / (image_path.stem + '.png')
                history_path = model_dir / 'scores' / (image_path.stem + '.json')
                history = json.loads(history_path.read_text())
                assert final.resolve() in written_images and final.is_file()
                with Image.open(final) as saved:
                    assert saved.size == size and saved.tobytes() == result.tobytes()
                assert 1 <= history['completed_optimization_steps'] <= config['optimization_steps']
                assert len(history['frames']) == history['completed_optimization_steps'] + 1
                if config['save_intermediates']:
                    for frame in history['frames']:
                        frame_path = model_dir / frame['image']
                        assert frame_path.resolve() in written_images
                        with Image.open(frame_path) as saved:
                            assert saved.size == size
                video_path = None
                if config['save_video']:
                    video = history['video']
                    video_path = model_dir / video['path']
                    with av.open(str(video_path)) as container:
                        assert container.streams.video[0].codec_context.name == 'h264'
                        assert sum(1 for _ in container.decode(video=0)) == video['frames']
                    assert video['show_scores'] == config['video_show_scores']
                assert all(parameter.grad is None for component in ('text_encoder', 'vae', 'unet', 'controlnet', 'ram', 'fidelity')
                           for parameter in getattr(bundle, component).parameters())
                first, last = history['frames'][0], history['frames'][-1]
                row.update(status='passed', iterations=history['completed_optimization_steps'],
                           input_iqa=first['prediction'], final_iqa=last['prediction'],
                           iqa_change=last['prediction'] - first['prediction'],
                           input_mapped=first['mapped_score'], final_mapped=last['mapped_score'],
                           final_image=str(final), video=str(video_path) if video_path else None,
                           scores=str(history_path), intermediate_images=len(history['frames']) if config['save_intermediates'] else 0)
                completed.add((model, image_path.name))
                print('BATCH_IMAGE_DONE', json.dumps(row), flush=True)
                return result
            except Exception as error:
                row.update(status='failed', error=str(error), error_type=type(error).__name__)
                traceback.print_exc()
                print('BATCH_IMAGE_FAILED', model, image_path.name, str(error), flush=True)
                return None
            finally:
                row['elapsed_seconds'] = round(time.monotonic() - image_started, 2)
                report['results'].append(row)
                report['completed_runs'] = len(completed)
                report['current'] = None
                save()

        diffusion.gen, diffusion._update_latents = checked_gen, tracked_update
        save()
        try:
            entry.main(config)
            verify_originals()
            assert hashes == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
            current = (ROOT / 'ms_liqe.pt').stat()
            assert [current.st_size, current.st_mtime_ns, current.st_ino] == weight_before
            report.update(status='passed' if len(completed) == expected else 'partial',
                          input_images_unchanged=True, original_project_and_weights_unchanged=True,
                          copied_weight_metadata_unchanged=True, shared_models_reused=True)
        except BaseException as error:
            report.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
            raise
        finally:
            diffusion.gen, diffusion._update_latents = real_gen, real_update
            save()
            print('BATCH_FINISHED', report['status'], report['completed_runs'], '/', expected, flush=True)
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
