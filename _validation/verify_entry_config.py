"""Verify entry configuration without loading model weights."""

import sys

sys.dont_write_bytecode = True

import ast
import contextlib
from copy import deepcopy
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import runpy
import tempfile
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    spec = importlib.util.spec_from_file_location('enhance_entry', ROOT / 'MAP_IQA.py')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    assert not {'torch', 'helper_functions', 'MAP_diffusion_latent', 'model_loader'} & sys.modules.keys()
    assert not (ROOT / 'MAP_enhance3.py').exists()
    help_output = io.StringIO()
    with patch.object(sys, 'argv', [str(ROOT / 'MAP_IQA.py'), '--help']), contextlib.redirect_stdout(help_output):
        try:
            runpy.run_path(str(ROOT / 'MAP_IQA.py'), run_name='__main__')
        except SystemExit as error:
            assert error.code == 0
        else:
            raise AssertionError('Help started enhancement')
    assert '--no-save-video' in help_output.getvalue()
    assert 'torch' not in sys.modules
    import model_config
    import ram
    assert 'torch' not in sys.modules
    assert ram.__all__ == []
    assert not {'ram.inference', 'ram.transform', 'ram.models'} & set(sys.modules)
    mapping_tree = ast.parse((ROOT / 'mapping2.py').read_text())
    parameter_node = next(node for node in mapping_tree.body if isinstance(node, ast.Assign) and node.targets[0].id == 'QUALITY_MAPPING_PARAMETERS')
    supported = tuple(ast.literal_eval(parameter_node.value))
    calls, factories, prompts, model_loads = [], [], [], []

    class FakeTensor:
        def __init__(self, prompt):
            self.prompt = prompt

        def to(self, target):
            return self

    class FakeTokenizer:
        model_max_length = 77

        def __call__(self, prompt, **kwargs):
            assert kwargs == {
                'padding': 'max_length', 'max_length': 77, 'truncation': True,
                'return_tensors': 'pt', 'return_overflowing_tokens': True,
            }
            prompts.append(prompt)
            return SimpleNamespace(input_ids=FakeTensor(prompt))

    def fake_gen(**kwargs):
        assert Path.cwd() == ROOT
        calls.append(kwargs)

    def fake_quality(model, **kwargs):
        factories.append((model, kwargs))
        return object()

    models = SimpleNamespace(
        tokenizer=FakeTokenizer(),
        text_encoder=lambda tokens: SimpleNamespace(last_hidden_state=FakeTensor(tokens.prompt)),
        device='cuda', dtype='float16',
    )

    def fake_load_models(**kwargs):
        model_loads.append(kwargs['config'])
        return models

    modules = {name: ModuleType(name) for name in ('mapping2', 'MAP_diffusion_latent', 'model_loader', 'torch', 'tqdm')}
    modules['mapping2'].SUPPORTED_QUALITY_MODELS = supported
    modules['MAP_diffusion_latent'].gen = fake_gen
    modules['model_loader'].load_models = fake_load_models
    modules['model_loader'].load_quality_scorer = fake_quality
    modules['torch'].autocast = lambda device: contextlib.nullcontext()
    modules['torch'].no_grad = contextlib.nullcontext
    modules['tqdm'].tqdm = lambda images, **kwargs: images

    with tempfile.TemporaryDirectory(prefix='diffusion-entry-config-', dir='/tmp') as scratch, patch.dict(sys.modules, modules), patch('output_manager.check_video_backend') as video_check:
        scratch = Path(scratch)
        inputs = scratch / 'inputs'
        inputs.mkdir()
        (inputs / 'one.png').write_bytes(b'')
        (inputs / 'two.JPG').write_bytes(b'')
        (inputs / 'notes.txt').write_text('not an image')
        (inputs / 'directory.png').mkdir()

        defaults_before = deepcopy(entry.CONFIG)
        parsed_defaults = entry.parse_config([])
        assert parsed_defaults == entry.CONFIG
        parsed_defaults['quality_models'].append('niqe')
        assert entry.CONFIG == defaults_before, 'Parsing exposed mutable CONFIG defaults'
        boolean_keys = (
            'save_intermediates', 'save_video', 'video_show_scores',
            'use_momentum', 'use_nesterov', 'tied_latents',
        )
        for key in boolean_keys:
            flag = key.replace('_', '-')
            assert entry.parse_config(['--' + flag])[key] is True
            assert entry.parse_config(['--no-' + flag])[key] is False
            assert entry.parse_config(['--no-' + flag, '--' + flag])[key] is True
            assert entry.parse_config(['--no-' + key])[key] is False
        assert entry.parse_config(['--clip-grad-val', 'None'])['clip_grad_val'] is None

        default = deepcopy(entry.CONFIG)
        default['input_dir'] = str(inputs)
        default['output_dir'] = str(scratch / 'default_outputs')
        with contextlib.redirect_stdout(io.StringIO()):
            entry.main(default)
        assert len(calls) == 2 and len(factories) == 1 and len(model_loads) == 1
        assert prompts == ['']
        assert factories[0][1]['config'] is model_loads[0]
        assert factories[0][1]['grad_scale'] == 1
        for args in calls:
            assert args['models'] is models
            assert args['diffusion_steps'] == default['diffusion_steps']
            assert args['optimization_steps'] == default['optimization_steps']
            assert args['fidelity_weight'] == default['fidelity_weight']
            assert args['latent_step_size'] == 2
            assert 'quality_lambda' not in args
            assert args['output_options'].save_intermediates is True
            assert args['output_options'].save_video is True
            assert args['output_options'].video_show_scores is True
            assert args['clip_grad_val'] == 1e-3
            assert args['mix_weight'] == 0.93 ** (50 / 20)
            assert args['seed'] == 0 and args['use_momentum'] is True
            assert args['use_nesterov'] is False and args['tied_latents'] is False
            assert args['quality_model'] == 'ms_liqe'
            assert args['output_dir'] == default['output_dir']
            assert not {'model_guidance_type', 'renormalize_latents', 'perturb_grad_scale', 'grad_scale', 'embedding_conditional'} & args.keys()

        calls.clear(); factories.clear(); prompts.clear(); model_loads.clear()
        custom_argv = [
            '--input-dir', str(inputs), '--output-dir', str(scratch / 'custom_outputs'),
            '--quality-models', 'ms_liqe', 'niqe', '--fidelity_weight', '37.5',
            '--no-save-intermediates', '--no-save-video', '--no-video-show-scores',
            '--diffusion-steps', '40', '--optimization-steps', '3', '--mix-weight-base', '0.9',
            '--grad-scale', '2', '--clip-grad-val', 'none', '--seed', '123',
            '--latent-step-size', '0.75', '--no-use-momentum', '--use-nesterov', '--tied-latents',
            '--text-prompt', 'a detailed photo',
        ]
        custom = entry.parse_config(custom_argv)
        assert set(custom) == set(entry.CONFIG)
        outside = scratch / 'outside'
        outside.mkdir()
        previous = Path.cwd()
        try:
            os.chdir(outside)
            with patch.object(sys, 'argv', [str(ROOT / 'MAP_IQA.py'), *custom_argv]), contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(ROOT / 'MAP_IQA.py'), run_name='__main__')
            assert Path.cwd() == outside
        finally:
            os.chdir(previous)
        assert len(calls) == 4 and len(factories) == 2 and len(model_loads) == 1
        assert prompts == ['a detailed photo']
        assert all(factory[1]['grad_scale'] == 2 for factory in factories)
        assert all(factory[1]['config'] is model_loads[0] for factory in factories)
        for args in calls:
            assert args['models'] is models
            assert args['fidelity_weight'] == 37.5
            assert args['latent_step_size'] == 0.75
            options = args['output_options']
            assert not options.save_intermediates and not options.save_video and not options.video_show_scores
            assert args['diffusion_steps'] == 40 and args['optimization_steps'] == 3
            assert args['clip_grad_val'] is None
            assert args['seed'] == 123 and args['tied_latents'] is True
            assert args['use_momentum'] is False and args['use_nesterov'] is True
            assert args['mix_weight'] == 0.9 ** (50 / 40)
            assert args['text_embedding'].prompt == 'a detailed photo'
        destinations = {
            (args['quality_model'], args['image_path'], args['output_dir'])
            for args in calls
        }
        assert len(destinations) == 4
        assert video_check.call_count == 1, 'Video backend checked while video output was disabled'
        assert 'quality_lambdas' not in entry.CONFIG
        assert entry.CONFIG == defaults_before, 'CLI overrides changed CONFIG'
        invalid_cli_cases = [
            ['--unknown-option', '1'], ['--fidelity-wei', '20'],
            ['--fidelity-weight'], ['--fidelity-weight', '-1'], ['--fidelity-weight', 'nan'],
            ['--optimization-steps', '0'], ['--diffusion-steps', '20.5'],
            ['--quality-models'], ['--quality-models', 'unsupported'],
            ['--quality-models', 'ms_liqe', 'ms_liqe'],
            ['--save-video', 'False'], ['--clip-grad-val', 'invalid'],
            ['--input-dir', str(scratch / 'missing')],
        ]
        for argv in invalid_cli_cases:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                try:
                    entry.parse_config(argv)
                except SystemExit as error:
                    assert error.code == 2
                else:
                    raise AssertionError(f'Invalid CLI arguments accepted: {argv}')
            assert 'error:' in stderr.getvalue()
        assert entry.parse_config(['--text-prompt', ''])['text_prompt'] == ''
        assert entry.parse_config(['--fidelity-weight', '0'])['fidelity_weight'] == 0
        (inputs / 'one.JPG').write_bytes(b'')
        try:
            entry.main(default)
        except ValueError as error:
            assert '重命名' in str(error)
        else:
            raise AssertionError('Conflicting output names accepted')
        finally:
            (inputs / 'one.JPG').unlink()
        assert entry.resolve_path('relative_outputs') == ROOT / 'relative_outputs'

        invalid_cases = [
            ('quality_models', []), ('quality_models', ['liqe_oppo_ms3']),
            ('quality_models', ['ms_liqe', 'ms_liqe']), ('quality_models', 'ms_liqe'),
            ('fidelity_weight', -1),
            ('fidelity_weight', math.nan), ('fidelity_weight', math.inf),
            ('fidelity_weight', True), ('fidelity_weight', '100'), ('diffusion_steps', 0),
            ('diffusion_steps', 1001), ('optimization_steps', 0),
            ('mix_weight_base', 0), ('mix_weight_base', 1.01),
            ('grad_scale', 0), ('clip_grad_val', -1), ('seed', 0.5),
            ('latent_step_size', 0), ('latent_step_size', -1),
            ('latent_step_size', math.nan), ('latent_step_size', math.inf),
            ('latent_step_size', True), ('latent_step_size', '2'),
            ('use_momentum', 1), ('text_prompt', None),
            ('save_intermediates', 1), ('save_video', None), ('video_show_scores', 'yes'),
            ('input_dir', str(scratch / 'missing')),
        ]
        for key, value in invalid_cases:
            config = deepcopy(default)
            config[key] = value
            try:
                entry.validate_config(config)
            except ValueError:
                pass
            else:
                raise AssertionError(f'Invalid configuration accepted: {key}={value!r}')

        zero_fidelity = deepcopy(default)
        zero_fidelity['fidelity_weight'] = 0
        entry.validate_config(zero_fidelity)
        assert entry.build_gen_kwargs(zero_fidelity)['fidelity_weight'] == 0

        module = modules['MAP_diffusion_latent']

        def fail_gen(*args, **kwargs):
            raise RuntimeError('simulated inference error')

        module.gen = fail_gen
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                entry.main(default)
        except RuntimeError as error:
            assert str(error) == 'simulated inference error'
        else:
            raise AssertionError('Inference failure was swallowed')
        assert Path.cwd() == previous

    report = {
        'status': 'passed', 'entry_import_does_not_load_weights': True,
        'model_config_and_RAM_root_import_without_torch': True,
        'default_gen_arguments_preserved': True, 'custom_parameters_forwarded': True,
        'model_outputs_do_not_collide': True, 'duplicate_image_stems_rejected': True,
        'video_and_intermediate_switches_forwarded': True, 'quality_lambda_removed': True, 'invalid_configs_rejected': len(invalid_cases),
        'working_directory_restored_after_success_and_failure': True,
        'shared_models_loaded_once_per_run': True,
        'fidelity_weight_default_and_override': True,
        'latent_step_size_default_and_override': True,
        'model_config_shared_with_loaders': True, 'model_weights_loaded': False,
        'renamed_entrypoint': 'MAP_IQA.py',
        'help_runs_without_torch_or_weights': True,
        'cli_defaults_match_CONFIG': True, 'cli_does_not_mutate_CONFIG': True,
        'cli_overrides_reach_main_and_gen': True,
        'cli_boolean_switches_and_underscore_aliases': True,
        'cli_none_disables_gradient_clipping': True,
        'invalid_cli_arguments_rejected': len(invalid_cli_cases),
    }
    target = ROOT / '_validation' / 'entry_config_result.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
