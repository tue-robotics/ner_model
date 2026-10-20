import os
import unittest
from types import SimpleNamespace
from unittest import mock

from ner_model.candidates import _candidate_text, candidates_from_common, load_candidates
from ner_model.linker import EntityCandidate


class TestCandidateText(unittest.TestCase):
    def test_candidate_text(self):
        self.assertEqual(_candidate_text("dinner_table", "dining room"), "dinner table, dining room")
        self.assertEqual(_candidate_text("dinner_table"), "dinner table")


class TestCandidatesFromCommon(unittest.TestCase):
    def test_candidates_from_common(self):
        # A small stand-in for robocup_knowledge's common.py
        common = SimpleNamespace(
            object_names=["coke"],
            objects=[{"name": "coke", "category": "drink"}, {"name": "coke", "category": "other"}],
            location_names=["dinner_table"],
            location_rooms=["kitchen"],
            locations=[{"name": "dinner_table", "room": "dining_room"}],
            names=["alex"],
        )
        self.assertEqual(
            candidates_from_common(common),
            [
                EntityCandidate(entity_id="coke", label="Object", text="coke, drink"),
                EntityCandidate(entity_id="dinner_table", label="Location", text="dinner table, dining room"),
                EntityCandidate(entity_id="kitchen", label="Location", text="kitchen"),
                EntityCandidate(entity_id="alex", label="Person", text="alex"),
            ],
        )

    def test_missing_attributes_give_no_candidates(self):
        self.assertEqual(candidates_from_common(SimpleNamespace()), [])


class TestLoadCandidates(unittest.TestCase):
    def test_no_robot_env_gives_no_candidates(self):
        with mock.patch.dict(os.environ, clear=True):
            self.assertEqual(load_candidates(), [])


if __name__ == "__main__":
    unittest.main()
