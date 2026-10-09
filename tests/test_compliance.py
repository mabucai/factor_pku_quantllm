import json
import unittest

from check_compliance import forbidden_path, inspect_content


class ComplianceTests(unittest.TestCase):
    def test_nested_nonempty_auth_is_detected_without_echo(self):
        marker = "synthetic_" + "test_marker"
        data = json.dumps({"connection": {"AuthCode": marker}}).encode()
        findings = inspect_content("settings.json", data)
        self.assertTrue(findings)
        self.assertNotIn(marker, str(findings))

    def test_python_literal_is_detected(self):
        data = " = ".join(("CTP_PASSWORD", repr("synthetic_" + "marker"))).encode()
        self.assertTrue(inspect_content("demo.py", data))

    def test_snapshot_detected(self):
        data = ("ACCOUNT " + "balance=" + str(123) + " frozen=" + str(4)).encode()
        self.assertTrue(inspect_content("output.txt", data))
        snapshot = dict(zip(("balance", "available"), (123, 100)))
        self.assertTrue(inspect_content("output.json", json.dumps(snapshot).encode()))

    def test_private_files_rejected_even_when_empty(self):
        for path in ("config.json", "config/ctp_settings.json", ".env", "logs/run.log", "snapshots/funds.csv", "secret.key", ".vntrader/ctp.json"):
            with self.subTest(path=path):
                self.assertTrue(forbidden_path(path))

    def test_empty_samples_and_research_metrics_allowed(self):
        self.assertEqual(inspect_content("config.example.json", json.dumps({"AuthCode": ""}).encode()), [])
        self.assertEqual(inspect_content("metrics.json", b'{"rank_ic": 0.2, "sharpe": 1.1}'), [])
        self.assertFalse(forbidden_path(".vntrader/.gitkeep"))

    def test_nonempty_example_fails_even_without_credentials(self):
        self.assertTrue(inspect_content("config.example.json", b'{"lab_path": "local/data"}'))


if __name__ == "__main__":
    unittest.main()
