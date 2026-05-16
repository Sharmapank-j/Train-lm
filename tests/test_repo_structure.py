import unittest
from pathlib import Path


class RepoStructureTests(unittest.TestCase):
    def test_core_directories_exist(self):
        root = Path(__file__).resolve().parents[1]
        for name in [
            "backend",
            "frontend",
            "trainer",
            "inference",
            "telegram",
            "scripts",
            "docker",
            "docs",
            "tests",
            "datasets",
            "models",
            "exports",
            "checkpoints",
            "logs",
        ]:
            self.assertTrue((root / name).exists(), f"missing {name}")


if __name__ == "__main__":
    unittest.main()
