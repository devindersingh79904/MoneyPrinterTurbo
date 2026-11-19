"""Shaping adapter layer for Arabic-script subtitles.

Arabic-script text (Arabic, Persian, Urdu, ...) must first select glyph forms by context
and join letters, then be laid out right-to-left before it displays correctly; vowel marks
must also be positioned on their base letters by OpenType rules. Pillow only shapes text
when built with Raqm, and common installs such as Windows lack Raqm, so letters come out
disconnected, reversed and with misplaced vowel marks.

Subtitles containing Arabic script are therefore measured and drawn by khatt (HarfBuzz
shaping + FreeType rasterization, bundled in the wheel with no system dependency); khatt
never calls Pillow's text layout, so the result is identical whether or not Pillow has
Raqm, and no double shaping occurs. Other languages still use MoviePy's native TextClip
and render exactly as before.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable

import khatt
from khatt.fonts import font_path as khatt_font_path
from khatt.video import TextClip as ShapedTextClip
from loguru import logger

# Unicode blocks related to Arabic script: basic, supplement, extended A/B and
# presentation forms A/B (excluding the U+FEFF byte order mark).
_ARABIC_SCRIPT_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x0870, 0x089F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFC),
)

# Direction marks: invisible, they only decide the paragraph direction of a line.
RLM = "‏"
LRM = "‎"

# None of the bundled project fonts contain Arabic glyphs. When the chosen font lacks them,
# switch to khatt's bundled Noto Naskh Arabic (SIL OFL 1.1), which covers Arabic, Persian
# and Urdu.
FALLBACK_FONT = "noto-naskh"


def _is_arabic_char(char: str) -> bool:
    # Only consider letters, marks and symbols (categories L/M/S): the Arabic comma, question
    # mark and Arabic-Indic digits are shared with Thaana (Dhivehi), Syriac and N'Ko, so they
    # cannot identify Arabic script.
    return unicodedata.category(char)[0] in "LMS" and any(
        start <= ord(char) <= end for start, end in _ARABIC_SCRIPT_RANGES
    )


def needs_shaping(text: str | None) -> bool:
    """Return whether the text contains Arabic script that needs shaping."""
    return any(_is_arabic_char(char) for char in text or "")


def is_rtl_paragraph(text: str) -> bool:
    """
    Determine paragraph direction by the first strong directional character (Unicode bidi rules
    P2/P3).
    """
    for char in text:
        direction = unicodedata.bidirectional(char)
        if direction in ("R", "AL"):
            return True
        if direction == "L":
            return False
    return False


def direction_mark(text: str) -> str:
    """Direction mark matching the paragraph direction of ``text``.

    khatt treats every line as its own paragraph and decides direction by the first strong
    directional character on the line. Automatically wrapped lines must keep the original
    paragraph direction, otherwise a line starting with English or digits has its word order
    reversed, so this mark is prepended to every line.
    """
    return RLM if is_rtl_paragraph(text) else LRM


def resolve_font(font_path: str, text: str) -> tuple[str, bool]:
    """Return the font file actually used to draw ``text`` and whether the fallback font was chosen.

    Fall back only when the chosen font lacks Arabic-script glyphs; if only other characters
    such as emoji are missing, keep the user's font and let khatt skip them with a warning as
    before.
    """
    try:
        missing = khatt.missing_characters(text, font_path)
    except (KeyError, ValueError) as exc:
        # Keep subtitles readable when the font file is missing or khatt cannot read it.
        logger.warning(f"cannot check subtitle font glyphs, using {FALLBACK_FONT}: {exc}")
        return str(khatt_font_path(FALLBACK_FONT)), True
    if any(_is_arabic_char(char) for char in missing):
        return str(khatt_font_path(FALLBACK_FONT)), True
    return font_path, False


def fallback_fonts(shaped_font: str, font_path: str) -> list[str]:
    """Fonts to use for characters ``shaped_font`` cannot draw in Arabic-script subtitles.

    When the fallback font is used, characters in the same subtitle that it lacks (for
    example Chinese after Arabic) are still drawn with the user's chosen font; khatt falls
    back per character run, so nothing is lost.
    """
    if not font_path or shaped_font == font_path:
        return []
    try:
        khatt_font_path(font_path)
    except (KeyError, ValueError):
        return []  # chosen font unreadable; resolve_font already warned
    return [font_path]


def video_font(font_path: str, subtitles: Iterable[str]) -> str:
    """Choose one font for all Arabic-script subtitles of a video.

    Choosing per subtitle could make subtitles of one video flip between two fonts, so the
    decision is made once over all subtitles; the font fallback and undrawable characters
    are each reported once.
    """
    text = "".join(subtitle for subtitle in subtitles if needs_shaping(subtitle))
    if not text:
        return font_path
    font, fell_back = resolve_font(font_path, text)
    if fell_back:
        logger.warning(
            f"subtitle font has no Arabic-script glyphs, using {FALLBACK_FONT} "
            f"for Arabic-script subtitles: {font_path}"
        )
    skipped = khatt.missing_characters(
        text, font, fallback_fonts=fallback_fonts(font, font_path)
    )
    if skipped:
        # E.g. emoji: neither the fallback nor the chosen font has these glyphs, so they are
        # skipped when drawing.
        logger.warning(
            "characters in Arabic-script subtitles that the subtitle font cannot "
            f"draw are skipped: {''.join(skipped[:20])}"
        )
    return font


def supports_text(font_path: str, text: str) -> bool:
    """
    Return whether the font actually drawing ``text`` (the fallback plus the chosen font when
    needed) can draw all of its characters.
    """
    font, _ = resolve_font(font_path, text)
    return not khatt.missing_characters(
        text, font, fallback_fonts=fallback_fonts(font, font_path)
    )


def text_width(
    text: str, font_path: str, font_size: int, fallbacks: Iterable[str] = ()
) -> int:
    """
    Measure the width (pixels) of one line using the actual shaped glyphs; see
    :func:`fallback_fonts` for ``fallbacks``.
    """
    left, _, right, _ = khatt.text_bbox(
        text, font=font_path, size_px=int(font_size), fallback_fonts=list(fallbacks)
    )
    return right - left


def text_clip(native, *, shaped_font: str | None = None, **kwargs):
    """Create a subtitle TextClip: Arabic-script text is drawn by khatt, the rest goes to
    ``native``.

    ``native`` is the MoviePy ``TextClip`` the caller uses; non-Arabic text is passed to it
    unchanged, exactly as before this change. The other arguments match MoviePy 2.2's
    ``TextClip``. ``shaped_font`` is the font for Arabic-script text (see
    :func:`video_font`); when omitted it is chosen from ``font`` and this text, and
    characters it cannot draw are drawn with ``font``.
    """
    text = kwargs.get("text")
    if not needs_shaping(text):
        return native(**kwargs)
    font = kwargs.get("font") or ""
    if shaped_font is None:
        shaped_font, _ = resolve_font(font, text)
    return ShapedTextClip(
        **{**kwargs, "font": shaped_font},
        fallback_fonts=fallback_fonts(shaped_font, font),
    )
