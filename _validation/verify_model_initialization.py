"""Verify import isolation, explicit loading, frozen models and failure propagation."""

import sys
sys.dont_write_bytecode = True

from contextlib import ExitStack
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from verify_one import install_write_guard, verify_originals

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def causes(error):
    result = []
    while error is not None:
        result.append(error)
        error = error.__cause__
    return result


def main():
    verify_originals()
    with tempfile.TemporaryDirectory(prefix='diffusion-model-init-', dir='/tmp') as name:
        scratch = Path(name)
        os.environ.update({
            'TMPDIR': name, 'MPLCONFIGDIR': str(scratch / 'matplotlib'),
            'KERAS_HOME': str(scratch / 'keras'), 'XDG_CACHE_HOME': str(scratch / 'xdg'),
            'CUDA_CACHE_DISABLE': '1', 'TORCHINDUCTOR_COMPILE_THREADS': '1',
            'TORCHINDUCTOR_CACHE_DIR': str(scratch / 'inductor'),
        })
        tempfile.tempdir = name
        install_write_guard(scratch)
        import torch
        from PIL import ImageFile, UnidentifiedImageError
        from error_context import error_context
        import model_loader

        # Importing utilities/the algorithm must neither load weights nor change global modes.
        initial_grad = torch.is_grad_enabled()
        initial_anomaly = torch.is_anomaly_enabled()
        initial_truncated = ImageFile.LOAD_TRUNCATED_IMAGES
        with ExitStack() as stack:
            for target in ('torch.load', 'torch.cuda._lazy_init', 'model_loader.load_models', 'model_loader.load_quality_scorer'):
                stack.enter_context(patch(target, side_effect=AssertionError(f'Import called {target}')))
            import helper_functions
            import MAP_diffusion_latent as enhancement
        assert torch.is_grad_enabled() == initial_grad
        assert torch.is_anomaly_enabled() == initial_anomaly
        assert ImageFile.LOAD_TRUNCATED_IMAGES == initial_truncated
        assert 'ms_liqe' not in sys.modules and 'ram.models.ram_lora' not in sys.modules
        assert not any(hasattr(helper_functions, name) for name in ('vae', 'unet', 'controlnet', 'ram_model', 'clip'))

        # Frozen parameters retain gradients with respect to the image input.
        linear = torch.nn.Linear(3, 2)
        expected = torch.randn(1, 3, requires_grad=True)
        reference = linear(expected).square().sum()
        expected_grad = torch.autograd.grad(reference, expected)[0]
        model_loader._prepare_model(linear, 'cpu')
        actual_input = expected.detach().clone().requires_grad_()
        linear(actual_input).square().sum().backward()
        assert torch.equal(expected_grad, actual_input.grad)
        assert not linear.training and all(not p.requires_grad and p.grad is None for p in linear.parameters())

        with patch('torch.cuda.is_available', return_value=False):
            try:
                model_loader.load_models()
            except RuntimeError as error:
                assert 'CUDA GPU' in str(error)
            else:
                raise AssertionError('Missing GPU accepted')
        try:
            model_loader._require_paths(scratch / 'missing-first', scratch / 'missing-second')
        except FileNotFoundError as error:
            assert 'missing-first' in str(error) and 'missing-second' in str(error)
        else:
            raise AssertionError('Missing weights accepted')

        loads = []
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.moves = []
            def to(self, device, **options):
                self.moves.append((device, options))
                return self
        def factory(component):
            def load(*args, **kwargs):
                loads.append((component, args, kwargs))
                return FakeModel()
            return load
        modules = {name: ModuleType(name) for name in (
            'transformers', 'diffusers', 'models', 'models.controlnet', 'models.unet_2d_condition',
            'ram', 'ram.models', 'torchvision', 'nlpd',
        )}
        for package in ('models', 'ram', 'ram.models'):
            modules[package].__path__ = []
        modules['transformers'].CLIPTextModel = SimpleNamespace(from_pretrained=factory('text encoder'))
        modules['transformers'].CLIPTokenizer = SimpleNamespace(from_pretrained=factory('tokenizer'))
        modules['diffusers'].AutoencoderKL = SimpleNamespace(from_pretrained=factory('VAE'))
        modules['models.unet_2d_condition'].UNet2DConditionModel = SimpleNamespace(from_pretrained_orig=factory('UNet'))
        modules['models.controlnet'].ControlNetModel = SimpleNamespace(from_pretrained=factory('ControlNet'))
        modules['ram.models'].ram = factory('RAM')
        modules['nlpd'].NLPD = factory('NLPD')
        modules['torchvision'].transforms = SimpleNamespace(
            Compose=lambda values: values, ToTensor=lambda: 'to tensor',
            Resize=lambda size: ('resize', size), Normalize=lambda **kwargs: kwargs,
        )
        previous = Path.cwd()
        try:
            os.chdir(scratch)
            with patch.dict(sys.modules, modules), patch('torch.cuda.is_available', return_value=True):
                bundle = model_loader.load_models()
            assert Path.cwd() == scratch
        finally:
            os.chdir(previous)
        assert [item[0] for item in loads] == ['text encoder', 'tokenizer', 'VAE', 'UNet', 'ControlNet', 'RAM', 'NLPD']
        for component in ('text_encoder', 'vae', 'unet', 'controlnet', 'ram', 'fidelity'):
            model = getattr(bundle, component)
            assert not model.training and all(not parameter.requires_grad for parameter in model.parameters())
            assert model.moves[0][0] == 'cuda'
        assert loads[0][1] == (str(ROOT.parent / 'stable-diffusion-2-base'),)
        assert loads[3][1] == (str(ROOT.parent / 'stable-diffusion-2-base'), str(ROOT / 'preset/models/seesr'))
        assert loads[5][2]['pretrained'] == str(ROOT / 'preset/models/ram_swin_large_14m.pth')
        assert loads[5][2]['pretrained_condition'] == str(ROOT / 'preset/DAPE.pth')
        assert bundle.fidelity.moves[0][1] == {}  # Preserve NLPD float32.

        # Override paths and loading options together, using the same fake loaders.
        custom_config = replace(
            model_loader.MODEL_CONFIG, sd_model_path=scratch / 'custom-sd',
            adapted_model_path=scratch / 'custom-adapted', ram_weight_path=scratch / 'custom-ram.pt',
            ram_condition_path=scratch / 'custom-condition.pt', device='cuda:1',
            diffusion_weight_dtype='float32', unet_use_image_cross_attention=True,
            ram_image_size=512, ram_vit='custom-test-backbone',
        )
        loads.clear()
        with patch.dict(sys.modules, modules), patch('torch.cuda.is_available', return_value=True), patch.object(
            model_loader, '_require_paths'
        ) as require_paths:
            custom_bundle = model_loader.load_models(config=custom_config)
        assert custom_bundle.device == 'cuda:1' and custom_bundle.dtype == torch.float32
        assert loads[0][1] == (str(custom_config.sd_model_path),)
        assert loads[3][1] == (str(custom_config.sd_model_path), str(custom_config.adapted_model_path))
        assert loads[3][2]['use_image_cross_attention'] is True
        assert loads[5][2] == {
            'pretrained': str(custom_config.ram_weight_path),
            'pretrained_condition': str(custom_config.ram_condition_path),
            'image_size': 512, 'vit': 'custom-test-backbone',
        }
        assert custom_bundle.ram_transform[0] == ('resize', (512, 512))
        assert require_paths.call_args.args[-2:] == (custom_config.ram_weight_path, custom_config.ram_condition_path)
        assert model_loader.MODEL_CONFIG.device == 'cuda'  # The shared defaults remain unchanged.

        def failure(*_, **__):
            raise OSError('simulated checkpoint failure')
        try:
            model_loader._load_component('UNet', 'known-weight-path', failure)
        except RuntimeError as error:
            assert all(text in str(error) for text in ('UNet', 'known-weight-path', 'simulated checkpoint failure'))
            assert isinstance(error.__cause__, OSError)
        else:
            raise AssertionError('Initialization error swallowed')
        fake_ms = ModuleType('ms_liqe')
        fake_ms.MSLIQE = failure
        with patch.dict(sys.modules, {'ms_liqe': fake_ms}), patch.object(model_loader, '_require_paths'):
            try:
                model_loader.load_quality_scorer('ms_liqe', device='cpu')
            except RuntimeError as error:
                assert 'ms_liqe.pt' in str(error) and isinstance(error.__cause__, OSError)
            else:
                raise AssertionError('Quality initialization error swallowed')
        for signal in (KeyboardInterrupt, SystemExit):
            try:
                with error_context('test operation'):
                    raise signal()
            except signal:
                pass
            else:
                raise AssertionError(f'{signal.__name__} swallowed')

        image = scratch / 'broken.png'
        image.write_bytes(b'not a valid image')
        args = dict(
            models=SimpleNamespace(fidelity=object()), image_path=image,
            quality_model='ms_liqe', quality_score_fn=lambda *_: None,
            text_embedding=object(), output_dir=scratch / 'outputs',
        )
        with patch.object(enhancement, '_build_diffusion_process', return_value=(None, None, None)):
            try:
                enhancement.gen(**args)
            except RuntimeError as error:
                chain = causes(error)
                assert isinstance(chain[-1], UnidentifiedImageError)
                assert 'Preparing image and RAM features' in str(error) and str(image) in str(error)
            else:
                raise AssertionError('Corrupt image accepted')
            with patch.object(enhancement, '_prepare_image', return_value=object()), patch.object(
                enhancement, '_initialize_latents', side_effect=OSError('inversion failed')
            ):
                try:
                    enhancement.gen(**args)
                except RuntimeError as error:
                    assert 'Initializing latents' in str(error) and isinstance(error.__cause__, OSError)
                else:
                    raise AssertionError('Inversion failure swallowed')
        # File handles close even when decoding raises.
        with image.open('rb') as handle:
            assert handle.read() == b'not a valid image'
        # Import real RAM exports without constructing a model or loading weights.
        with patch('torch.load', side_effect=AssertionError('RAM import loaded weights')), patch(
            'torch.cuda._lazy_init', side_effect=AssertionError('RAM import initialized CUDA')
        ):
            from ram.models import RAMLora, ram
            import ram.models as ram_models
        assert RAMLora.__module__ == 'ram.models.ram_lora'
        assert ram.__module__ == 'ram.models.ram_lora'
        assert ram_models.__all__ == ['RAMLora', 'ram']
        assert not {'ram.inference', 'ram.transform', 'ram.models.ram', 'ram.models.tag2text', 'ram.models.bert'} & set(sys.modules)
        verify_originals()
        assert torch.is_grad_enabled() == initial_grad
        report = {
            'status': 'passed', 'imports_load_no_weights_or_CUDA': True,
            'imports_preserve_grad_anomaly_and_PIL_modes': True,
            'frozen_parameters_preserve_input_gradients': True,
            'explicit_model_load_order_and_weight_paths': True,
            'all_models_eval_and_frozen': True, 'NLPD_dtype_preserved': True,
            'loading_independent_of_working_directory': True,
            'custom_model_config_forwarded_to_all_loaders': True,
            'RAM_exports_only_LoRA_without_weights_or_CUDA': True,
            'missing_CUDA_and_paths_rejected': True,
            'component_and_quality_load_errors_keep_context_and_cause': True,
            'corrupt_image_and_inversion_errors_keep_context_and_cause': True,
            'keyboard_interrupt_and_system_exit_propagate': True,
            'original_project_and_weights_unchanged': True,
        }
        (ROOT / '_validation/model_initialization_test_result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
