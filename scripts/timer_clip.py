"""The task timer's clock as a video, counting down to 00:00 one frame a second.

PowerPoint has no running clock, so the timer slides play this clip from when they
show. It holds its last frame, 00:00, once it ends. The poster is its first frame,
which the editor and a PDF show in its place.

``scripts/render_assets.py`` writes both into ``assets/``, and the deck embeds those
committed files, so building the deck needs no encoder. The digits are drawn in the
deck's Calibri Bold, read from the folders ``deck_layout`` searches.
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from deck_kit import BACKGROUND, CLOCK_SIZE, INK

ASSETS = Path(__file__).resolve().parents[1] / "assets"
CLIP = ASSETS / "timer_clock.mp4"
POSTER = ASSETS / "timer_clock.png"

PIXELS_PER_INCH = 300
SIZE = (
    round(CLOCK_SIZE[0] * PIXELS_PER_INCH),
    round(CLOCK_SIZE[1] * PIXELS_PER_INCH),
)
"""Width and height in pixels, both multiples of 16 so the encoder keeps them."""
POINTS = 32
FONT_FILE = "calibrib.ttf"


def frame(seconds_left: int, fonts: Path) -> Image.Image:
    """The clock at ``seconds_left``, as mm:ss in ink, right-aligned on the slide."""
    image = Image.new("RGB", SIZE, f"#{BACKGROUND}")
    face = ImageFont.truetype(
        str(fonts / FONT_FILE), round(POINTS / 72 * PIXELS_PER_INCH)
    )
    minutes, seconds = divmod(seconds_left, 60)
    ImageDraw.Draw(image).text(
        (SIZE[0], SIZE[1] / 2),
        f"{minutes:02d}:{seconds:02d}",
        font=face,
        fill=f"#{INK}",
        anchor="rm",
    )
    return image


def write(minutes: int, fonts: Path) -> None:
    """Encode the clip from ``minutes`` to 00:00, and save its first frame."""
    import imageio_ffmpeg  # the deck extra, needed only to re-encode

    total = minutes * 60
    frame(total, fonts).save(POSTER)
    encoder = imageio_ffmpeg.write_frames(
        str(CLIP),
        SIZE,
        fps=1,
        codec="libx264",
        quality=None,
        ffmpeg_log_level="error",
        output_params=[
            # No B-frames: Windows ignores the edit list that offsets them, which
            # at one frame a second would put the clock two seconds early.
            "-crf", "18", "-tune", "stillimage", "-profile:v", "main", "-bf", "0",
            "-movflags", "+faststart", "-threads", "1",
            "-fflags", "+bitexact", "-map_metadata", "-1",
        ],
    )  # fmt: skip
    encoder.send(None)
    for left in range(total, -1, -1):
        encoder.send(frame(left, fonts).tobytes())
    encoder.close()
