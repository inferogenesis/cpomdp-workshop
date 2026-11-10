import numpy as np
import pytest

pytest.importorskip("imageio_ffmpeg", reason="the deck extra encodes the timer clip")

import imageio_ffmpeg
from PIL import Image

import build_deck
import deck_layout
import timer_clip

MINUTES = pytest.mark.parametrize("minutes", build_deck.TIMER_MINUTES)

# Reading the clip starts ffmpeg as a subprocess, which JAX, imported by other tests,
# warns about. The child execs at once without running Python, so it cannot deadlock.
pytestmark = pytest.mark.filterwarnings(
    "ignore:os.fork\\(\\) was called:RuntimeWarning"
)


def test_there_is_a_clip_for_the_fifteen_and_the_twenty_minute_timer():
    assert build_deck.TIMER_MINUTES == (15, 20)


@MINUTES
def test_the_clip_runs_one_frame_a_second_from_the_full_time_to_zero(minutes):
    clip = timer_clip.clip(minutes)
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(clip))
    assert frames == minutes * 60 + 1
    assert seconds == pytest.approx(minutes * 60 + 1, abs=0.05)


@pytest.fixture(scope="module")
def fonts():
    measure = deck_layout.metrics()
    if measure is None:
        pytest.skip("the clock is drawn in Calibri, and its font file is missing")
    return measure.directory


def error(decoded, expected):
    return float(np.abs(decoded.astype(int) - np.asarray(expected, int)).mean())


@pytest.mark.slow
@MINUTES
def test_each_frame_shows_the_time_left_at_that_second(minutes, fonts):
    total = minutes * 60
    reader = imageio_ffmpeg.read_frames(str(timer_clip.clip(minutes)))
    meta = next(reader)
    width, height = meta["size"]
    assert (width, height) == timer_clip.SIZE
    checked = {0, 1, 59, 60, 61, 450, 899, total}
    for index, raw in enumerate(reader):
        if index not in checked:
            continue
        left = total - index
        decoded = np.frombuffer(raw, np.uint8).reshape(height, width, 3)
        shown = error(decoded, timer_clip.frame(left, fonts))
        neighbour = error(decoded, timer_clip.frame(left - 1 if left else 1, fonts))
        assert shown < 3, f"{left} s left"
        assert shown < neighbour / 2, f"{left} s left"
        checked.discard(index)
    assert not checked


@MINUTES
def test_the_poster_is_the_full_time(minutes, fonts):
    poster = Image.open(timer_clip.poster(minutes)).convert("RGB")
    assert np.array_equal(
        np.asarray(poster), np.asarray(timer_clip.frame(minutes * 60, fonts))
    )
