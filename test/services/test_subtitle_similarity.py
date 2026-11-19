import sys
import unittest
from pathlib import Path

# Allow running this file directly from the repository root.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import subtitle


class TestSubtitleSimilarityBoundaries(unittest.TestCase):
    """correct() uses similarity() to merge adjacent subtitles, so the result must always be defined.
    """

    def test_two_empty_strings_are_identical(self):
        self.assertEqual(subtitle.similarity("", ""), 1.0)

    def test_one_empty_string_has_zero_similarity(self):
        self.assertEqual(subtitle.similarity("", "hello"), 0.0)
        self.assertEqual(subtitle.similarity("hello", ""), 0.0)

    def test_case_folding_that_grows_text_stays_within_range(self):
        # U+0130 (LATIN CAPITAL LETTER I WITH DOT ABOVE) case-folds to "i" + U+0307,
        # so its length grows from 1 to 2; the old implementation used the pre-fold
        # length as the denominator and produced a negative similarity.
        self.assertEqual(len("İ".lower()), 2)
        for left, right in (
            ("İ", ""),
            ("", "İ"),
            ("İ", "i"),
            ("İstanbul", "istanbul"),
        ):
            with self.subTest(left=left, right=right):
                score = subtitle.similarity(left, right)
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

    def test_existing_matching_behaviour_is_unchanged(self):
        self.assertEqual(subtitle.similarity("Hello", "hello"), 1.0)
        self.assertEqual(subtitle.similarity("same", "same"), 1.0)
        self.assertLess(subtitle.similarity("hello", "world"), 0.5)

    def test_similarity_is_symmetric_for_case_folded_input(self):
        self.assertEqual(subtitle.similarity("İ", "i"), subtitle.similarity("i", "İ"))


if __name__ == "__main__":
    unittest.main()
