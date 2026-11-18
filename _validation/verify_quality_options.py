"""Check supported model routing, calibration, gradients and rejected names."""

import sys

sys.dont_write_bytecode = True

import ast
import contextlib
from dataclasses import replace
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import tempfile
from types import ModuleType
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXPECTED = (
    'ms_liqe', 'dbcnn', 'liqe', 'hyperiqa', 'musiq',
    'clipiqa+', 'niqe', 'paq2piq', 'unique',
)


def main():
    spec = importlib.util.spec_from_file_location('single_image_validation', ROOT / '_validation' / 'verify_one.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    before = json.loads((ROOT / '_validation' / 'cleanup_stage2_before.json').read_text())['files']

    with tempfile.TemporaryDirectory(prefix='diffusion-quality-options-', dir='/tmp') as scratch:
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        os.environ['TMPDIR'] = scratch
        os.environ['MPLCONFIGDIR'] = str(Path(scratch) / 'matplotlib')
        os.environ['CUDA_CACHE_DISABLE'] = '1'
        tempfile.tempdir = scratch
        verifier.install_write_guard(Path(scratch))
        import torch

        current = runpy.run_path(str(ROOT / 'mapping2.py'))
        old = {}
        exec(compile(before['mapping2.py'], '<mapping-before-cleanup>', 'exec'), old)
        assert current['SUPPORTED_QUALITY_MODELS'] == EXPECTED

        import model_loader
        routes = []
        for name in EXPECTED:
            legacy_name = 'liqe_oppo_ms3' if name == 'ms_liqe' else name
            values = torch.tensor([-10.0, 0.0, 0.5, 1.0, 5.0, 50.0, 100.0], requires_grad=True)
            mapped = current['logistic_mapping'](values, name)
            reference = old['logistic_mapping'](values, legacy_name)
            assert torch.equal(mapped, reference), name
            actual_gradient = torch.autograd.grad(mapped.sum(), values)[0]
            reference_gradient = torch.autograd.grad(reference.sum(), values)[0]
            assert torch.equal(actual_gradient, reference_gradient), name

            calls = []

            class FakeMetric:
                def __init__(self):
                    self.eval_called = False
                    self.device = None
                    self.frozen = False

                def requires_grad_(self, value):
                    self.frozen = not value
                    return self

                def eval(self):
                    self.eval_called = True
                    return self

                def to(self, device):
                    self.device = device
                    return self

                def __call__(self, image):
                    return image.mean()

            def metric_factory(model_name, **kwargs):
                metric = FakeMetric()
                calls.append((model_name, kwargs, metric))
                return metric

            ms_module, pyiqa_module = ModuleType('ms_liqe'), ModuleType('pyiqa')
            ms_module.MSLIQE = lambda **kwargs: metric_factory('ms_liqe', **kwargs)
            pyiqa_module.create_metric = metric_factory
            with patch.dict(sys.modules, {'ms_liqe': ms_module, 'pyiqa': pyiqa_module}), patch.object(
                model_loader, '_require_paths'
            ), contextlib.redirect_stdout(io.StringIO()):
                loss_fn = model_loader.load_quality_scorer(name, device='cpu', grad_scale=3)
                image = torch.linspace(-0.5, 0.5, 48).reshape(1, 3, 4, 4).requires_grad_()
                evaluation = loss_fn(image, name)
                loss = evaluation.value
            assert len(calls) == 1, calls
            expected_route = 'liqe_mix' if name == 'liqe' else name
            assert calls[0][0] == expected_route, calls
            if name == 'ms_liqe':
                assert calls[0][1] == {'ckpt': str(ROOT / 'ms_liqe.pt'), 'device': 'cpu', 'scale': [1, 0.75]}
                assert calls[0][2].eval_called
            else:
                assert calls[0][1]['as_loss'] is True
            if name == 'liqe':
                assert calls[0][1]['num_patch'] == 15
            if name == 'hyperiqa':
                assert calls[0][1]['pretrained_model_path'] == '/home/ps/.cache/torch/hub/checkpoints/HyperIQA-resnet50-koniq10k-48579ec9.pth'
            assert calls[0][2].device == 'cpu'
            assert calls[0][2].eval_called and calls[0][2].frozen
            if name != 'ms_liqe':
                assert calls[0][1]['device'] == 'cpu'
            prediction = ((image / 2) + 0.5).clamp(0, 1).float().mean()
            reference_loss = old['logistic_mapping']((-1 if name == 'niqe' else 1) * prediction, legacy_name).half() * 3
            assert torch.equal(loss, reference_loss), name
            assert evaluation.prediction == prediction.item()
            assert evaluation.mapped_score == current['logistic_mapping']((-1 if name == 'niqe' else 1) * prediction, name).item()
            loss.backward()
            assert torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0, name
            assert (image.grad < 0).all() if name == 'niqe' else (image.grad > 0).all()
            routes.append({'option': name, 'factory': calls[0][0], 'mapping_and_gradient_unchanged': True})

        custom_config = replace(
            model_loader.MODEL_CONFIG, ms_liqe_weight_path=Path(scratch) / 'custom-ms.pt',
            ms_liqe_scales=(1, 0.5), liqe_num_patch=9,
            hyperiqa_weight_path=Path(scratch) / 'custom-hyper.pt',
        )
        for option in ('ms_liqe', 'liqe', 'hyperiqa'):
            calls.clear()
            with patch.dict(sys.modules, {'ms_liqe': ms_module, 'pyiqa': pyiqa_module}), patch.object(
                model_loader, '_require_paths'
            ):
                model_loader.load_quality_scorer(option, device='cpu', config=custom_config)
            if option == 'ms_liqe':
                assert calls[0][1]['ckpt'] == str(custom_config.ms_liqe_weight_path)
                assert calls[0][1]['scale'] == [1, 0.5]
            elif option == 'liqe':
                assert calls[0][1]['num_patch'] == 9
            else:
                assert calls[0][1]['pretrained_model_path'] == str(custom_config.hyperiqa_weight_path)

        removed = {
            node.comparators[0].value
            for node in ast.walk(ast.parse(before['mapping2.py']))
            if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
            and node.left.id == 'model' and isinstance(node.comparators[0], ast.Constant)
        } - set(EXPECTED)
        removed.update({'brisque', 'ilniqe', 'nima', 'tres', 'hybrid', 'unknown'})
        for name in sorted(removed):
            for function in (
                lambda: model_loader.load_quality_scorer(name, device='cpu'),
                lambda: current['logistic_mapping'](torch.tensor(0.5), name),
            ):
                try:
                    function()
                except ValueError:
                    pass
                else:
                    raise AssertionError(f'Removed option accepted: {name}')

        class Rename(ast.NodeTransformer):
            def visit_Name(self, node):
                if node.id == 'LIQE_ms':
                    node.id = 'MSLIQE'
                return node

        original_class = next(node for node in ast.parse(before['LIQE.py']).body if isinstance(node, ast.ClassDef) and node.name == 'LIQE_ms')
        original_class.name = 'MSLIQE'
        original_class = Rename().visit(original_class)
        current_class = next(node for node in ast.parse((ROOT / 'ms_liqe.py').read_text()).body if isinstance(node, ast.ClassDef))
        assert ast.dump(original_class, include_attributes=False) == ast.dump(current_class, include_attributes=False)
        verifier.verify_originals()
        report = {
            'status': 'passed',
            'supported_options': list(EXPECTED),
            'factory_routing_and_mapping_checks': routes,
            'rejected_options': sorted(removed),
            'multi_scale_model_math_unchanged': True,
            'other_metric_weights_loaded': False,
            'all_metrics_frozen_eval_and_explicit_device': True,
            'custom_quality_model_config_forwarded': True,
            'original_project_and_weights_unchanged': True,
        }
        target = ROOT / '_validation' / 'quality_options_result.json'
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
