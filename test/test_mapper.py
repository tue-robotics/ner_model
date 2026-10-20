import unittest

from ner_model.linker import LinkResult
from ner_model.mapper import build_semantics, extract_spans, normalize_entity


def tagged(*pairs):
    """Model output for a sentence, given as (token, slot) pairs, wrapped in [CLS] ... [SEP]"""
    tokens = [("[CLS]", "O")] + list(pairs) + [("[SEP]", "O")]
    return [{"token": token, "slot": slot} for token, slot in tokens]


class FakeLinker:
    """Stands in for BiEncoderLinker, which needs a sentence-transformers model"""

    def __init__(self, entity_id=None, error=None):
        self._entity_id = entity_id
        self._error = error

    def link(self, slot_label, mention):
        if self._error:
            raise self._error
        if self._entity_id is None:
            return None
        return LinkResult(
            entity_id=self._entity_id, score=0.9, mention=mention, slot_label=slot_label, matched_text=self._entity_id
        )


class TestExtractSpans(unittest.TestCase):
    def test_merges_bio_tags(self):
        predictions = tagged(
            ("go", "B-Action-navigate-to"),
            ("to", "I-Action-navigate-to"),
            ("the", "O"),
            ("kitchen", "B-Location"),
        )
        self.assertEqual(extract_spans(predictions), [("Action-navigate-to", "go to"), ("Location", "kitchen")])

    def test_joins_word_pieces(self):
        predictions = tagged(("liv", "B-Location"), ("##ing", "I-Location"), ("room", "I-Location"))
        self.assertEqual(extract_spans(predictions), [("Location", "living room")])

    def test_keeps_underscore_inside_span(self):
        predictions = tagged(("dinner", "B-Location"), ("_", "O"), ("table", "I-Location"))
        self.assertEqual(extract_spans(predictions), [("Location", "dinner _ table")])

    def test_single_letter_prefix_joins_next_piece(self):
        predictions = tagged(("a", "O"), ("##lex", "B-Person"))
        self.assertEqual(extract_spans(predictions), [("Person", "alex")])

    def test_ignores_padding(self):
        predictions = tagged(("kitchen", "B-Location"), ("[PAD]", "[PAD]"))
        self.assertEqual(extract_spans(predictions), [("Location", "kitchen")])


class TestNormalizeEntity(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_entity("Living Room"), "living_room")
        self.assertEqual(normalize_entity(" kitchen _ cabinet "), "kitchen_cabinet")


class TestBuildSemantics(unittest.TestCase):
    def test_entities_attach_to_preceding_action(self):
        predictions = tagged(
            ("go", "B-Action-navigate-to"),
            ("to", "I-Action-navigate-to"),
            ("the", "O"),
            ("kitchen", "B-Location"),
            ("and", "O"),
            ("grab", "B-Action-pick-up"),
            ("a", "O"),
            ("coke", "B-Object"),
        )
        self.assertEqual(
            build_semantics(predictions),
            {
                "actions": [
                    {"action": "navigate-to", "target-location": {"id": "kitchen"}},
                    {"action": "pick-up", "object": {"type": "coke"}},
                ]
            },
        )

    def test_slot_to_semantics(self):
        cases = [
            ("SourceLocation", {"source-location": {"id": "kitchen"}}),
            ("TargetLocation", {"target-location": {"id": "kitchen"}}),
            ("Location", {"target-location": {"id": "kitchen"}}),
            ("Person", {"target-location": {"type": "person", "id": "kitchen"}}),
        ]
        for slot, expected in cases:
            with self.subTest(slot=slot):
                predictions = tagged(("go", "B-Action-navigate-to"), ("kitchen", "B-" + slot))
                self.assertEqual(build_semantics(predictions), {"actions": [dict(action="navigate-to", **expected)]})

    def test_area_is_skipped(self):
        predictions = tagged(("go", "B-Action-navigate-to"), ("kitchen", "B-Area"))
        self.assertEqual(build_semantics(predictions), {"actions": [{"action": "navigate-to"}]})

    def test_underscore_span_becomes_entity_id(self):
        predictions = tagged(
            ("go", "B-Action-navigate-to"), ("dinner", "B-Location"), ("_", "O"), ("table", "I-Location")
        )
        self.assertEqual(
            build_semantics(predictions),
            {"actions": [{"action": "navigate-to", "target-location": {"id": "dinner_table"}}]},
        )

    def test_entity_before_action_gives_unknown_action(self):
        predictions = tagged(("coke", "B-Object"))
        self.assertEqual(build_semantics(predictions), {"actions": [{"action": "unknown", "object": {"type": "coke"}}]})

    def test_place_without_object_gets_reference(self):
        predictions = tagged(("put", "B-Action-place"), ("it", "O"), ("on", "O"), ("the", "O"), ("table", "B-Location"))
        self.assertEqual(
            build_semantics(predictions),
            {"actions": [{"action": "place", "target-location": {"id": "table"}, "object": {"type": "reference"}}]},
        )

    def test_no_action_gives_unknown(self):
        self.assertEqual(build_semantics(tagged(("hello", "O"))), {"actions": [{"action": "unknown"}]})

    def test_linker_result_is_used(self):
        predictions = tagged(("go", "B-Action-navigate-to"), ("dining", "B-Location"), ("table", "I-Location"))
        self.assertEqual(
            build_semantics(predictions, linker=FakeLinker(entity_id="dinner_table")),
            {"actions": [{"action": "navigate-to", "target-location": {"id": "dinner_table"}}]},
        )

    def test_linker_miss_or_error_falls_back_to_normalized_text(self):
        expected = {"actions": [{"action": "navigate-to", "target-location": {"id": "dining_table"}}]}
        predictions = tagged(("go", "B-Action-navigate-to"), ("dining", "B-Location"), ("table", "I-Location"))
        for linker in (FakeLinker(entity_id=None), FakeLinker(error=RuntimeError("model crashed"))):
            with self.subTest(linker=linker):
                self.assertEqual(build_semantics(predictions, linker=linker), expected)


if __name__ == "__main__":
    unittest.main()
