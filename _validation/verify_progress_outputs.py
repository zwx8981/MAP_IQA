"""Exercise output switches, original-size PNGs, real H.264 videos and failure cleanup."""

import sys
sys.dont_write_bytecode = True

import itertools
import json
from fractions import Fraction
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageDraw
import av

from verify_one import install_write_guard, verify_originals

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from output_manager import EnhancementOutput, OutputOptions, _render_video_frame, check_video_backend


def evaluation(prediction):
    return SimpleNamespace(value=SimpleNamespace(item=lambda: prediction * 20),
                           prediction=prediction, mapped_score=prediction * 20)


def main():
    verify_originals()
    check_video_backend()
    cases = []
    with tempfile.TemporaryDirectory(prefix='diffusion-output-validation-', dir='/tmp') as directory:
        scratch = Path(directory)
        tempfile.tempdir = directory
        install_write_guard(scratch)
        images = [Image.new('RGB', (321, 193), color) for color in ((35, 50, 70), (90, 115, 130), (155, 180, 195))]
        for image in images:
            ImageDraw.Draw(image).rectangle((15, 15, 70, 55), fill=(245, 85, 55))
        for save_frames, save_video, show_scores in itertools.product((False, True), repeat=3):
            root = scratch / f'{save_frames}-{save_video}-{show_scores}'
            options = OutputOptions(
                save_intermediates=save_frames, save_video=save_video, video_show_scores=show_scores,
            )
            options.validate()
            with EnhancementOutput(root, 'ms_liqe', 'sample.png', 'input/sample.png', options) as output:
                temp_frames = output.frame_dir
                for step, image in enumerate(images):
                    output.add_frame(image, evaluation(3 + step * 0.5), step)
                output.finish(images[-1])
            final = root / 'ms_liqe/final/sample.png'
            with Image.open(final) as result:
                assert result.size == (321, 193) and result.tobytes() == images[-1].tobytes()
            history = json.loads((root / 'ms_liqe/scores/sample.json').read_text())
            assert [row['step'] for row in history['frames']] == [0, 1, 2]
            assert [row['prediction'] for row in history['frames']] == [3, 3.5, 4]
            frames_dir = root / 'ms_liqe/intermediate/sample'
            if save_frames:
                assert len(list(frames_dir.glob('*.png'))) == 3
                with Image.open(frames_dir / '000_input.png') as original:
                    assert original.tobytes() == images[0].tobytes()
            else:
                assert not frames_dir.exists()
                assert all(row['image'] is None for row in history['frames'])
                if temp_frames is not None:
                    assert not temp_frames.exists(), 'Video temporary frames were not cleaned'
            video = root / 'ms_liqe/videos/sample.mp4'
            if save_video:
                assert video.is_file() and history['video']['show_scores'] == show_scores
                with av.open(str(video)) as container:
                    stream = container.streams.video[0]
                    assert stream.codec_context.name == 'h264'
                    assert stream.width % 2 == 0 and stream.height % 2 == 0
                    decoded = list(container.decode(video=0))
                    assert len(decoded) == history['video']['frames'] == 2
                    assert stream.average_rate == Fraction(2, 7)
                    assert stream.duration * stream.time_base == history['video']['duration_seconds'] == 7
                    assert container.duration == 7 * av.time_base
                    for index, (frame, expected) in enumerate(zip(decoded, images[1:])):
                        assert frame.pts * frame.time_base == index * Fraction(7, 2)
                        # H.264 is lossy. Compare a point inside the displayed image with a color tolerance.
                        expected_rgb = expected.getpixel((120, 100))
                        y = (stream.height - expected.height) // 2 + 100
                        actual_rgb = frame.to_image().getpixel((120, y))
                        assert all(abs(a - b) < 6 for a, b in zip(actual_rgb, expected_rgb))
                    if show_scores:
                        assert stream.width == 642 and stream.height == 480
                    else:
                        assert stream.width == 322 and stream.height == 194
            else:
                assert not video.exists() and history['video'] is None
            cases.append({'save_intermediates': save_frames, 'save_video': save_video,
                          'video_show_scores': show_scores, 'passed': True})

        # Without scores, the renderer contains the image unchanged and only codec padding.
        plain = _render_video_frame(images[1], history['frames'], 1, 'ms_liqe', False)
        assert plain.crop((0, 0, 321, 193)).tobytes() == images[1].tobytes()
        # A shorter new run cannot leave old generated steps in this image's intermediate folder.
        root = scratch / 'rerun'
        options = OutputOptions(save_video=False)
        with EnhancementOutput(root, 'ms_liqe', 'sample.png', 'input.png', options) as output:
            (output.frame_dir / '999_step.png').write_bytes(b'old frame')
            (output.frame_dir / 'notes.txt').write_text('keep this user file')
        with EnhancementOutput(root, 'ms_liqe', 'sample.png', 'input.png', options) as output:
            output.add_frame(images[0], evaluation(3), 0)
            output.add_frame(images[-1], evaluation(4), 1)
            output.finish(images[-1])
        assert not (root / 'ms_liqe/intermediate/sample/999_step.png').exists()
        assert (root / 'ms_liqe/intermediate/sample/notes.txt').read_text() == 'keep this user file'

        # Encoding errors preserve the final PNG and error history, and retain an older valid video.
        root = scratch / 'failed'
        old_video = root / 'ms_liqe/videos/sample.mp4'
        old_video.parent.mkdir(parents=True)
        old_video.write_bytes(b'previous successful video')
        with EnhancementOutput(root, 'ms_liqe', 'sample.png', 'input.png', OutputOptions(save_intermediates=False)) as output:
            temp_frames = output.frame_dir
            output.add_frame(images[0], evaluation(3), 0)
            output.add_frame(images[-1], evaluation(4), 1)
            with patch('output_manager._encode_video', side_effect=OSError('simulated encoder failure')):
                try:
                    output.finish(images[-1])
                except RuntimeError as error:
                    assert isinstance(error.__cause__, OSError) and 'Encoding progression video' in str(error)
                else:
                    raise AssertionError('Encoder failure swallowed')
        assert not temp_frames.exists()
        assert old_video.read_bytes() == b'previous successful video'
        assert len(list(old_video.parent.iterdir())) == 1, 'Partial encoded video was not cleaned'
        failed = json.loads((root / 'ms_liqe/scores/sample.json').read_text())
        assert 'simulated encoder failure' in failed['video_error']
        assert (root / 'ms_liqe/final/sample.png').is_file()
        verify_originals()
        report = {'status': 'passed', 'switch_combinations': cases,
                  'real_h264_encode_decode': True, 'score_panel_optional': True,
                  'odd_size_PNGs_preserved': True, 'codec_dimensions_even_without_cropping': True,
                  'original_input_preserved_in_score_history': True,
                  'video_starts_at_step_one_and_lasts_seven_seconds': True,
                  'temporary_frames_cleaned': True, 'stale_generated_frames_removed': True,
                  'unrelated_files_retained': True, 'encoding_failure_preserves_final_and_previous_video': True,
                  'original_project_and_weights_unchanged': True}
        (ROOT / '_validation/progress_output_test_result.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
