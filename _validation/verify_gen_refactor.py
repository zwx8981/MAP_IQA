"""CPU regression checks against the saved pre-refactor implementation.

Load definitions via AST so these checks do not initialize models or weights.
The objective and optimizer references come directly from the original gen body.
"""

import sys
sys.dont_write_bytecode = True

import ast
import copy
from contextlib import nullcontext
from dataclasses import dataclass
import inspect
import itertools
import json
import math
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace

from verify_one import install_write_guard, verify_originals

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from error_context import error_context
from model_loader import QualityEvaluation
from output_manager import EnhancementOutput, OutputOptions, check_video_backend


def execute(nodes, namespace):
    module = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(module, '<isolated regression definitions>', 'exec'), namespace)


def reference_function(name, arguments, body, returned):
    function = ast.parse(f'def {name}({arguments}):\n    pass').body[0]
    function.body = copy.deepcopy(body) + [ast.Return(value=ast.parse(returned, mode='eval').body)]
    return function


def main():
    verify_originals()
    with tempfile.TemporaryDirectory(prefix='diffusion-gen-regression-', dir='/tmp') as name:
        scratch = Path(name)
        os.environ.update({
            'TMPDIR': name, 'CUDA_CACHE_DISABLE': '1',
            'TORCHINDUCTOR_COMPILE_THREADS': '1',
            'TORCHINDUCTOR_CACHE_DIR': str(scratch / 'inductor'),
            'TRITON_CACHE_DIR': str(scratch / 'triton'),
        })
        tempfile.tempdir = name
        install_write_guard(scratch)
        import numpy as np
        import torch
        from torch import nn
        from PIL import Image

        old_source = json.loads((ROOT / '_validation/gen_refactor_before.json').read_text())['files']['MAP_diffusion_latent.py']
        old_tree = ast.parse(old_source)
        new_tree = ast.parse((ROOT / 'MAP_diffusion_latent.py').read_text())
        base = dict(
            torch=torch, nn=nn, Image=Image, np=np, dataclass=dataclass,
            error_context=error_context, EnhancementModels=SimpleNamespace,
            EnhancementOutput=EnhancementOutput, OutputOptions=OutputOptions, check_video_backend=check_video_backend,
            math=math, Path=Path, random=random, device='cpu', torch_dtype=torch.float32,
            autocast=lambda *_: nullcontext(), tqdm=lambda values, **_: values,
            SUPPORTED_QUALITY_MODELS=(
                'ms_liqe', 'dbcnn', 'liqe', 'hyperiqa', 'musiq',
                'clipiqa+', 'niqe', 'paq2piq', 'unique',
            ),
        )
        helpers = ast.parse((ROOT / 'helper_functions.py').read_text())
        execute([node for node in helpers.body if isinstance(node, ast.FunctionDef) and node.name in {
            'get_alpha_and_beta', 'forward_step', 'reverse_step',
            'crop_to_original', 'prep_image_for_return',
        }], base)
        new = dict(base)
        execute([node for node in new_tree.body if
                 isinstance(node, (ast.FunctionDef, ast.ClassDef)) or
                 isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id.isupper()
                                                     for target in node.targets)], new)
        old = dict(base)
        execute([node for node in old_tree.body if isinstance(node, ast.ClassDef)
                 and node.name in {'SteppingLayer', 'MixingLayer'}], old)
        old_gen = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == 'gen')
        assignments = sorted([
            node for node in ast.walk(old_gen) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in {'loss', 'score'} for target in node.targets)
        ], key=lambda node: node.lineno)
        assert len(assignments) == 5
        execute([reference_function(
            'objective', 'im, source_im, quality_lambda, loss_fn, model_guidance_dict, fidelity_fn',
            assignments, '(loss, score)',
        )], old)
        loop = next(node for node in ast.walk(old_gen) if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == 'm')
        start = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == 'grad' for target in node.targets))
        stop = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Name) and node.test.id == 'renormalize_latents')
        execute([reference_function(
            'update', 'orig_latent_pair, prev_b_arr, m, clip_grad_val, use_momentum, use_nesterov, tied_latents',
            loop.body[start:stop], 'latent_pair',
        )], old)
        for function in ('MixingLayer',):
            before = next(node for node in old_tree.body if getattr(node, 'name', '') == function)
            after = next(node for node in new_tree.body if getattr(node, 'name', '') == function)
            assert ast.dump(before) == ast.dump(after), function

        torch.manual_seed(17)
        source = torch.randn(1, 3, 4, 4)
        candidate = torch.randn_like(source)
        def quality(image, model):
            value = (image.square().mean() * 1.7 + 0.25).half().view(1)
            return QualityEvaluation(value, value.item(), value.item())
        def fidelity(image, reference):
            return (image - reference).square().mean().view(1)
        for trial in range(5):
            candidate = torch.randn_like(source)
            left = candidate.clone().requires_grad_()
            right = candidate.clone().requires_grad_()
            expected, score = old['objective'](left, source, 1, lambda *args: quality(*args).value, {'quality_model_str': 'ms_liqe'}, fidelity)
            actual, new_score = new['_compute_objective'](
                right, source, quality, 'ms_liqe', fidelity,
                image_path='input.png', iteration=0, candidate_index=0,
            )
            assert torch.equal(expected, actual) and score == new_score.value.item(), trial
            expected.backward()
            actual.backward()
            assert torch.equal(left.grad, right.grad), trial

        class ParameterizeFidelity(ast.NodeTransformer):
            def visit_Constant(self, node):
                if type(node.value) is int and node.value == 100:
                    return ast.copy_location(ast.Name(id='fidelity_weight', ctx=ast.Load()), node)
                return node
        configurable = reference_function(
            'configured_objective',
            'im, source_im, quality_lambda, loss_fn, model_guidance_dict, fidelity_fn, fidelity_weight',
            assignments, '(loss, score)',
        )
        execute([ParameterizeFidelity().visit(configurable)], old)
        fidelity_cases = 0
        for fidelity_weight, sample in itertools.product((0, 1, 37.5, 100, 250), (0, 1)):
            left, right = candidate.clone().requires_grad_(), candidate.clone().requires_grad_()
            expected, _ = old['configured_objective'](
                left, source, 1, lambda *args: quality(*args).value, {'quality_model_str': 'ms_liqe'}, fidelity, fidelity_weight,
            )
            actual, _ = new['_compute_objective'](
                right, source, quality, 'ms_liqe', fidelity,
                fidelity_weight=fidelity_weight, image_path='input.png', iteration=0, candidate_index=0,
            )
            assert torch.equal(expected, actual)
            expected.backward()
            actual.backward()
            assert torch.equal(left.grad, right.grad)
            fidelity_cases += 1

        update_cases = 0
        for momentum, nesterov, tied, clip in itertools.product((False, True), (False, True), (False, True), (None, 1e-3)):
            previous_old, previous_new = [None, None], [None, None]
            for iteration in range(3):
                latents = torch.randn(2, 4, 4, 4)
                grad = torch.randn_like(latents)
                left, right = latents.clone().requires_grad_(), latents.clone().requires_grad_()
                left.grad, right.grad = grad.clone(), grad.clone()
                expected = old['update'](left, previous_old, iteration, clip, momentum or nesterov, nesterov, tied)
                actual = new['_update_latents'](right, previous_new, iteration, clip, momentum or nesterov, nesterov, tied)
                assert torch.equal(expected, actual), (momentum, nesterov, tied, clip, iteration)
                for a, b in zip(previous_old, previous_new):
                    assert (a is None and b is None) or torch.equal(a, b)
                update_cases += 1

        # Explicit examples: half-scaled gradients are [2, -3], with no clipping/momentum.
        for step, expected_values in ((0.25, [0.5, -0.75]), (4, [8.0, -12.0])):
            latents = torch.zeros(2, 1, 1, 1, requires_grad=True)
            latents.grad = torch.tensor([4.0, -6.0]).reshape_as(latents)
            actual = new['_update_latents'](
                latents, [None, None], 0, None, False, False, False, latent_step_size=step,
            )
            assert torch.equal(actual.detach().flatten(), torch.tensor(expected_values)), step

        class Schedule:
            def __init__(self):
                self.config = SimpleNamespace(num_train_timesteps=1000)
                self.alphas_cumprod = torch.linspace(0.99, 0.01, 1000)
                self.final_alpha_cumprod = self.alphas_cumprod[0]
            def set_timesteps(self, count):
                self.num_inference_steps = count
                self.timesteps = torch.tensor([999, 500, 0])
        embedding, ram, control = torch.ones(1, 1), torch.ones(1, 1), torch.ones(1, 3, 4, 4)
        class Control(nn.Module):
            def forward(self, latents, time, **kwargs):
                assert kwargs['encoder_hidden_states'] is embedding
                assert kwargs['image_encoder_hidden_states'] is ram
                assert kwargs['controlnet_cond'] is control
                return [latents * 0.03], latents * 0.04
        class UNet(nn.Module):
            def forward(self, latents, time, **kwargs):
                assert kwargs['encoder_hidden_states'] is embedding
                assert kwargs['image_encoder_hidden_states'] is ram
                return (latents * 0.125 + kwargs['down_block_additional_residuals'][0]
                        + kwargs['mid_block_additional_residual'],)
        before_step = old['SteppingLayer'](UNet(), Control(), embedding, None, scheduler=Schedule(), num_timesteps=3)
        after_step = new['SteppingLayer'](UNet(), Control(), embedding, scheduler=Schedule(), num_timesteps=3)
        stepping_cases = 0
        for reverse, index, time in itertools.product((False, True), range(3), (999, 500, 0)):
            inputs = torch.randn(2, 4, 4, 4)
            left, right = inputs.clone().requires_grad_(), inputs.clone().requires_grad_()
            args = (torch.tensor([float(index)]), torch.tensor([float(time)]))
            expected = before_step(*args, left, ram, control, reverse=reverse)
            actual = after_step(*args, right, ram, control, reverse=reverse)
            assert all(torch.equal(a, b) for a, b in zip(expected, actual)), (reverse, index, time)
            expected[2].square().sum().backward()
            actual[2].square().sum().backward()
            assert torch.equal(left.grad, right.grad), (reverse, index, time)
            stepping_cases += 1

        def failure(*_):
            raise ValueError('scorer failed')
        try:
            new['_compute_objective'](candidate, source, failure, 'ms_liqe', fidelity,
                                      image_path='broken.png', iteration=2, candidate_index=1)
        except RuntimeError as error:
            assert isinstance(error.__cause__, ValueError)
            assert all(value in str(error) for value in ('ms_liqe', 'broken.png', 'step 2', 'candidate 1'))
        else:
            raise AssertionError('Scoring failure was swallowed')
        def interruption(*_):
            raise KeyboardInterrupt()
        try:
            new['_compute_objective'](candidate, source, interruption, 'ms_liqe', fidelity,
                                      image_path='input.png', iteration=0, candidate_index=0)
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError('KeyboardInterrupt was swallowed')

        for invalid in (math.nan, math.inf):
            try:
                new['_compute_objective'](
                    candidate, source, lambda *_: QualityEvaluation(torch.tensor([invalid]), invalid, invalid), 'ms_liqe', fidelity,
                    image_path='invalid.png', iteration=1, candidate_index=0,
                )
            except RuntimeError as error:
                assert isinstance(error.__cause__, FloatingPointError)
            else:
                raise AssertionError('Non-finite quality score accepted')
            latent = torch.zeros(2, 4, 4, 4, requires_grad=True)
            latent.grad = torch.full_like(latent, invalid)
            try:
                new['_update_latents'](latent, [None, None], 0, 1e-3, True, False, False)
            except FloatingPointError:
                pass
            else:
                raise AssertionError('Non-finite latent gradient accepted')

        try:
            new['_compute_objective'](
                candidate, source, quality, 'ms_liqe',
                lambda *_: torch.tensor([math.nan]),
                image_path='invalid-fidelity.png', iteration=2, candidate_index=1,
            )
        except FloatingPointError as error:
            assert 'invalid-fidelity.png' in str(error) and 'step 2' in str(error)
        else:
            raise AssertionError('Non-finite objective accepted')
        try:
            new['_update_latents'](torch.zeros(2, 4, 4, 4, requires_grad=True),
                                   [None, None], 0, 1e-3, True, False, False)
        except RuntimeError as error:
            assert 'No latent gradient' in str(error)
        else:
            raise AssertionError('Missing latent gradient accepted')

        python_state, numpy_state = random.getstate(), np.random.get_state()
        torch_state = torch.get_rng_state()
        def random_input_quality(*_):
            assert not torch.is_grad_enabled()
            random.random(); np.random.rand(); torch.rand(1)
            return QualityEvaluation(torch.tensor([1.0]), 1.0, 1.0)
        new['_evaluate_input'](source, random_input_quality, 'ms_liqe')
        assert random.getstate() == python_state and torch.equal(torch.get_rng_state(), torch_state)
        current_numpy = np.random.get_state()
        assert numpy_state[0] == current_numpy[0] and np.array_equal(numpy_state[1], current_numpy[1])
        assert numpy_state[2:] == current_numpy[2:]

        # Exercise the complete orchestration using differentiable CPU stand-ins.
        class VAE:
            dtype = torch.float32
            def encode(self, image):
                encoded = torch.cat([image, image[:, :1]], dim=1)
                return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda **_: encoded))
            def decode(self, latent):
                return SimpleNamespace(sample=latent[:, :3])
        class PipelineStep:
            def __call__(self, i, t, latents, *_):
                return i, t, latents * 0.9
            def inverse(self, i, t, latents, *_):
                return i, t, latents / 0.9
        class Distance:
            def to(self, *_):
                return self
            def __call__(self, *values):
                return fidelity(*values)
        models = SimpleNamespace(device='cpu', vae=VAE(), fidelity=Distance())
        new['_prepare_image'] = lambda *_: new['PreparedImage'](source, source * 0.5 + 0.5, ram, (0, 0, 0, 0), Image.new('RGB', (4, 4)))
        new['_build_diffusion_process'] = lambda *_: (PipelineStep(), new['MixingLayer'](), torch.tensor([1, 0]))
        real_cuda_seed, real_cuda_seed_all = torch.cuda.manual_seed, torch.cuda.manual_seed_all
        torch.cuda.manual_seed = lambda *_: None
        torch.cuda.manual_seed_all = lambda *_: None
        try:
            image_path = scratch / 'input.png'
            Image.new('RGB', (4, 4)).save(image_path)
            arguments = dict(models=models, image_path=image_path, quality_model='ms_liqe', quality_score_fn=quality,
                             text_embedding=embedding, output_dir=scratch / 'normal', optimization_steps=3,
                             latent_step_size=0.75, output_options=OutputOptions(save_video=False))
            for invalid_step in (0, -1, math.nan, math.inf, True, '2'):
                try:
                    new['gen'](**dict(arguments, latent_step_size=invalid_step))
                except ValueError as error:
                    assert 'latent_step_size' in str(error)
                else:
                    raise AssertionError('Invalid step size accepted')
            real_update = new['_update_latents']
            received_step_sizes = []
            def checked_update(*args, **kwargs):
                received_step_sizes.append(kwargs['latent_step_size'])
                return real_update(*args, **kwargs)
            new['_update_latents'] = checked_update
            result = new['gen'](**arguments)
            assert isinstance(result, Image.Image) and result.size == (4, 4)
            assert received_step_sizes == [0.75] * 3
            assert (scratch / 'normal/ms_liqe/final/input.png').is_file()
            assert len(list((scratch / 'normal/ms_liqe/intermediate/input').glob('*.png'))) == 4
            scores, updates = [], []
            def high_quality(image, model):
                scores.append(model)
                return QualityEvaluation(image.mean().view(1) * 0 + 100, 100, 100)
            def counted_update(*args, **kwargs):
                updates.append(1)
                assert kwargs['latent_step_size'] == 0.75
                return real_update(*args, **kwargs)
            new['_update_latents'] = counted_update
            result = new['gen'](**dict(arguments, quality_score_fn=high_quality, output_dir=scratch / 'early'))
            assert isinstance(result, Image.Image)
            assert len(list((scratch / 'early/ms_liqe/intermediate/input').glob('*.png'))) == 2
            assert len(scores) == 3 and len(updates) == 1  # Finish the triggering round before stopping.
            try:
                new['gen'](**dict(arguments, quality_score_fn=failure, output_dir=scratch / 'failed'))
            except RuntimeError as error:
                causes = []
                while error is not None:
                    causes.append(error)
                    error = error.__cause__
                assert isinstance(causes[-1], ValueError)
                assert any('Scoring input image' in str(cause) for cause in causes)
            else:
                raise AssertionError('gen did not propagate a scoring failure')
            class FailingOutput(EnhancementOutput):
                def add_frame(self, image, evaluation, step):
                    if step > 0:
                        raise PermissionError('simulated output write error')
                    return super().add_frame(image, evaluation, step)
            new['EnhancementOutput'] = FailingOutput
            try:
                new['gen'](**arguments)
            except RuntimeError as error:
                messages = []
                while error.__cause__ is not None:
                    messages.append(str(error))
                    error = error.__cause__
                assert isinstance(error, PermissionError)
                assert any(str(image_path) in message for message in messages)
                assert any('Optimization step 0' in message for message in messages)
            else:
                raise AssertionError('Save failure swallowed')
        finally:
            torch.cuda.manual_seed, torch.cuda.manual_seed_all = real_cuda_seed, real_cuda_seed_all
        parameters = inspect.signature(new['gen']).parameters
        assert 'quality_lambda' not in parameters
        assert len(parameters) == 18 and all(value.kind == inspect.Parameter.KEYWORD_ONLY for value in parameters.values())
        verify_originals()
        report = {
            'status': 'passed', 'objective_values_and_gradients': 5,
            'configurable_fidelity_values_and_gradients': fidelity_cases,
            'optimizer_update_cases': update_cases,
            'custom_step_size_numeric_examples': 2, 'step_size_forwarded_through_gen': True,
            'invalid_step_sizes_rejected_by_gen': True, 'coupled_step_values_and_gradients': stepping_cases,
            'mixing_layer_ast_unchanged': True, 'quality_lambda_removed_and_default_objective_preserved': True,
            'input_IQA_evaluation_preserves_random_states': True,
            'gen_pipeline_and_output_layout': True, 'early_stop_finishes_current_round': True,
            'scoring_error_context_and_cause': True, 'save_error_context_and_cause': True, 'keyboard_interrupt_propagates': True, 'non_finite_values_rejected': True,
            'public_keyword_only_parameters': len(parameters),
            'original_code_resources_and_weights_unchanged': True,
        }
        (ROOT / '_validation/gen_regression_result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
