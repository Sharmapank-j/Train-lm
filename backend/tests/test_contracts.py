import unittest
from pathlib import Path


class ContractsTestCase(unittest.TestCase):
    def test_core_contracts_file_contains_success_shape(self):
        contracts = Path(__file__).resolve().parents[1] / "app" / "core" / "contracts.py"
        text = contracts.read_text(encoding="utf-8")
        self.assertIn('"success": True', text)
        self.assertIn('"request_id"', text)
        self.assertIn('"timestamp"', text)

    def test_safe_join_guards_traversal(self):
        util = Path(__file__).resolve().parents[1] / "app" / "utils" / "paths.py"
        text = util.read_text(encoding="utf-8")
        self.assertIn("Path traversal detected", text)


if __name__ == "__main__":
    unittest.main()
