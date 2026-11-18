"""Save final images, optional progression frames, score history and H.264 videos."""

from dataclasses import dataclass
from fractions import Fraction
import json
import re
from pathlib import Path
import tempfile

from PIL import Image, ImageDraw, ImageFont

from error_context import error_context


VIDEO_DURATION_SECONDS = 7


@dataclass(frozen=True)
class OutputOptions:
    save_intermediates: bool = True
    save_video: bool = True
    video_show_scores: bool = True

    def validate(self):
        for name in ('save_intermediates', 'save_video', 'video_show_scores'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be True or False')


def check_video_backend():
    with error_context('Checking H.264 video encoder (activate the iqa environment)'):
        import av
        av.codec.Codec('libx264', 'w')


def _font(size):
    try:
        return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size)
    except OSError:
        return ImageFont.load_default()


def _render_video_frame(image, history, index, model_name, show_scores):
    """Add a separate score panel; the enhanced PNG itself is never annotated."""
    preview = image.convert('RGB')
    preview.thumbnail((960, 720), Image.Resampling.LANCZOS)
    width, height = preview.size
    if show_scores:
        height = max(480, height)
    canvas_width = width + (320 if show_scores else 0)
    canvas = Image.new('RGB', (canvas_width + canvas_width % 2, height + height % 2), '#10151e')
    canvas.paste(preview, (0, (canvas.height - preview.height) // 2))
    if not show_scores:
        return canvas

    draw = ImageDraw.Draw(canvas)
    left = width + 24
    current = history[index]
    lower_better = model_name == 'niqe'
    draw.text((left, 24), model_name, font=_font(24), fill='#e8eef5')
    stage = f'Step {current["step"]} / {history[-1]["step"]}'
    draw.text((left, 64), stage, font=_font(18), fill='#9dadc2')
    draw.text((left, 106), 'IQA prediction', font=_font(16), fill='#9dadc2')
    draw.text((left, 130), f'{current["prediction"]:.4f}', font=_font(34), fill='#65dcc5')
    direction = 'lower is better' if lower_better else 'higher is better'
    draw.text((left, 176), direction, font=_font(14), fill='#9dadc2')
    draw.text((left, 214), f'Mapped score: {current["mapped_score"]:.2f} / 100', font=_font(15), fill='#e8eef5')
    delta = current['prediction'] - history[0]['prediction']
    draw.text((left, 242), f'Change from input: {delta:+.4f}', font=_font(15), fill='#e8eef5')

    x0, y0, x1, y1 = left + 8, canvas.height - 150, canvas.width - 28, canvas.height - 44
    values = [row['prediction'] for row in history[1:]]
    low, high = min(values), max(values)
    margin = max((high - low) * 0.12, abs(high) * 0.005, 0.001)
    low, high = low - margin, high + margin
    draw.line((x0, y0, x0, y1, x1, y1), fill='#607187', width=1)
    draw.text((x0, y0 - 24), f'{high:.3f}', font=_font(12), fill='#9dadc2')
    draw.text((x0, y1 + 10), 'Step 1', font=_font(12), fill='#9dadc2')
    draw.text((x1 - 32, y1 + 10), 'Final', font=_font(12), fill='#9dadc2')
    points = [
        (x0 + i / max(1, len(values) - 1) * (x1 - x0), y1 - (value - low) / (high - low) * (y1 - y0))
        for i, value in enumerate(values[:index])
    ]
    if len(points) > 1:
        draw.line(points, fill='#65dcc5', width=3)
    px, py = points[-1]
    draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill='#f5c86b')
    return canvas


def _encode_video(path, frame_paths, history, model_name, options):
    import av

    # Step 0 is the original input kept in the score history and intermediate PNGs.
    step_paths = frame_paths[1:]
    if not step_paths:
        raise ValueError('Video export requires at least one optimization image')
    fps = Fraction(len(step_paths), VIDEO_DURATION_SECONDS)
    time_base = 1 / fps
    with av.open(str(path), 'w', format='mp4', options={'movflags': '+faststart'}) as container:
        stream = container.add_stream('libx264', rate=fps)
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'medium'}
        for index, frame_path in enumerate(step_paths, start=1):
            with Image.open(frame_path) as image:
                rendered = _render_video_frame(image, history, index, model_name, options.video_show_scores)
            if index == 1:
                stream.width, stream.height = rendered.size
            frame = av.VideoFrame.from_image(rendered)
            frame.pts = index - 1
            frame.time_base = time_base
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        return {'frames': len(step_paths), 'fps': float(fps), 'size': [stream.width, stream.height],
                'duration_seconds': VIDEO_DURATION_SECONDS, 'codec': 'h264'}


class EnhancementOutput:
    """Own only this run's frame list; videos never glob older experiment files."""

    def __init__(self, output_dir, quality_model, output_name, input_path, options):
        self.root = Path(output_dir) / quality_model
        self.model = quality_model
        self.name = output_name
        self.stem = Path(output_name).stem
        self.input_path = str(input_path)
        self.options = options
        self.history = []
        self.frame_paths = []
        self._temporary = None
        self.final_path = self.root / 'final' / output_name
        self.history_path = self.root / 'scores' / f'{self.stem}.json'
        self.video_path = self.root / 'videos' / f'{self.stem}.mp4'

    def __enter__(self):
        if self.options.save_intermediates:
            self.frame_dir = self.root / 'intermediate' / self.stem
            self.frame_dir.mkdir(parents=True, exist_ok=True)
            # Replace only generated frames for this image; keep unrelated files.
            for previous in self.frame_dir.iterdir():
                if previous.is_file() and re.fullmatch(r'(000_input|[0-9]{3,}_step)\.png', previous.name):
                    previous.unlink()
        elif self.options.save_video:
            self._temporary = tempfile.TemporaryDirectory(prefix='diffusion-video-frames-')
            self.frame_dir = Path(self._temporary.name)
        else:
            self.frame_dir = None
        return self

    def __exit__(self, *exc):
        if self._temporary is not None:
            self._temporary.cleanup()

    def add_frame(self, image, evaluation, step):
        record = {'step': step, 'stage': 'input' if step == 0 else 'optimization',
                  'prediction': evaluation.prediction, 'mapped_score': evaluation.mapped_score,
                  'optimization_score': evaluation.value.item(), 'image': None}
        if self.frame_dir is not None:
            name = '000_input.png' if step == 0 else f'{step:03d}_step.png'
            path = self.frame_dir / name
            image.save(path)
            self.frame_paths.append(path)
            if self.options.save_intermediates:
                record['image'] = str(path.relative_to(self.root))
        self.history.append(record)

    def _write_history(self, video=None, error=None):
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        record = {'input_image': self.input_path, 'quality_model': self.model,
                  'prediction_higher_is_better': self.model != 'niqe',
                  'score_space': 'padded optimization image',
                  'completed_optimization_steps': len(self.history) - 1,
                  'final_image': str(self.final_path.relative_to(self.root)),
                  'video': video, 'frames': self.history}
        if error is not None:
            record['video_error'] = str(error)
        temporary = self.history_path.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temporary.replace(self.history_path)

    def finish(self, image):
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(self.final_path)
        self._write_history()
        if self.options.save_video:
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=self.video_path.parent, suffix='.mp4', delete=False) as file:
                temporary = Path(file.name)
            try:
                with error_context(f'Encoding progression video for {self.input_path}'):
                    info = _encode_video(temporary, self.frame_paths, self.history, self.model, self.options)
                temporary.replace(self.video_path)
            except Exception as error:
                self._write_history(error=error)
                raise
            finally:
                temporary.unlink(missing_ok=True)
            self._write_history(video={'path': str(self.video_path.relative_to(self.root)),
                                       'show_scores': self.options.video_show_scores, **info})
