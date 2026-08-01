"""Regression tests for scripts/validate_public_repo.py.

The validator is executed end-to-end in CI (python scripts/validate_public_repo.py).
These unit tests pin down the obsolete-repository-reference detection so a
renamed repository cannot silently slip through again.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "scripts" / "validate_public_repo.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_public_repo", VALIDATOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ValidatorObsoleteNameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = _load_validator()
        cls.obsolete_values = (
            (cls.mod.OBSOLETE_URL, cls.mod.CANONICAL_URL),
            (cls.mod.OBSOLETE_NAME, cls.mod.CANONICAL_NAME),
        )

    def test_detects_obsolete_name(self):
        violations = self.mod._collect_violations(
            ["# " + self.mod.OBSOLETE_NAME],
            Path("README.md"),
            self.obsolete_values,
        )
        self.assertEqual(len(violations), 1)
        location, found, canonical = violations[0]
        self.assertEqual(location, "README.md:1")
        self.assertEqual(found, self.mod.OBSOLETE_NAME)
        self.assertEqual(canonical, self.mod.CANONICAL_NAME)

    def test_detects_obsolete_url(self):
        violations = self.mod._collect_violations(
            ["git clone " + self.mod.OBSOLETE_URL + ".git"],
            Path("README.md"),
            self.obsolete_values,
        )
        self.assertEqual(len(violations), 1)
        location, found, canonical = violations[0]
        self.assertEqual(found, self.mod.OBSOLETE_URL)
        self.assertEqual(canonical, self.mod.CANONICAL_URL)

    def test_detects_multiple_lines(self):
        violations = self.mod._collect_violations(
            [
                "clean line",
                self.mod.OBSOLETE_NAME,
                "another clean line",
                self.mod.OBSOLETE_URL,
            ],
            Path("NOTICE.md"),
            self.obsolete_values,
        )
        self.assertEqual([location for location, _, _ in violations], [
            "NOTICE.md:2",
            "NOTICE.md:4",
        ])

    def test_ignores_historical_references(self):
        violations = self.mod._collect_violations(
            ["The project was formerly known as " + self.mod.OBSOLETE_NAME + "."],
            Path("NOTICE.md"),
            self.obsolete_values,
        )
        self.assertEqual(violations, [])

    def test_canonical_values_are_not_flagged(self):
        violations = self.mod._collect_violations(
            [
                "git clone " + self.mod.CANONICAL_URL + ".git",
                self.mod.CANONICAL_NAME,
            ],
            Path("README.md"),
            self.obsolete_values,
        )
        self.assertEqual(violations, [])

    def test_identity_check_passes_on_clean_tree(self):
        # The canonical files in this repository are clean, so the identity
        # check must pass without raising SystemExit.
        self.mod.validate_canonical_identity()


if __name__ == "__main__":
    unittest.main()
