import unittest
from unittest.mock import patch

from app.__main__ import main


class EntryPointTests(unittest.TestCase):
    def test_startup_failure_is_sanitized_without_traceback(self) -> None:
        with patch(
            "sys.argv", ["typace", "--satellite-catalog", r"C:\Users\alice\secret.json"]
        ), patch(
            "app.__main__.TyPaceApp",
            side_effect=RuntimeError(
                r"failed C:\Users\alice\secret.json alice@example.com token=abc"
            ),
        ):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(
            str(raised.exception),
            "failed <redacted-path> <redacted-email> token=<redacted>",
        )


if __name__ == "__main__":
    unittest.main()
