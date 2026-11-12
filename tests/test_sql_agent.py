import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import evaluate
from build_db import build_database
from libs.ollama_engine import run_agent_turn
from tools.calculator import calculate
from tools.sql_tools import execute_sql, generate_sql, get_db_schema


class SqlAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "store.db"
        build_database(self.db_path)
        self.db_path_patch = patch("tools.sql_tools.DB_PATH", self.db_path)
        self.db_path_patch.start()

    def tearDown(self) -> None:
        self.db_path_patch.stop()
        self.temp_dir.cleanup()

    def test_seeded_database_has_four_tables_and_over_300_rows(self) -> None:
        schema = json.loads(get_db_schema())
        self.assertEqual(
            set(schema), {"customers", "products", "orders", "order_items"}
        )
        result = json.loads(
            execute_sql(
                "SELECT "
                "(SELECT COUNT(*) FROM customers) + "
                "(SELECT COUNT(*) FROM products) + "
                "(SELECT COUNT(*) FROM orders) + "
                "(SELECT COUNT(*) FROM order_items)"
            )
        )
        self.assertGreater(result["rows"][0][0], 300)

    def test_database_build_is_deterministic(self) -> None:
        first = json.loads(execute_sql("SELECT * FROM order_items ORDER BY id"))
        build_database(self.db_path)
        second = json.loads(execute_sql("SELECT * FROM order_items ORDER BY id"))
        self.assertEqual(first, second)

    def test_sql_tools_allow_reads_and_reject_writes_and_multiple_statements(self) -> None:
        self.assertEqual(generate_sql(" SELECT COUNT(*) FROM customers;"),
                         "SELECT COUNT(*) FROM customers")
        self.assertEqual(json.loads(execute_sql("SELECT 1"))["rows"], [[1]])
        for sql in (
            "DELETE FROM customers",
            "WITH x AS (SELECT 1) DELETE FROM customers",
            "SELECT 1; SELECT 2",
        ):
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                generate_sql(sql)

    def test_calculator_supports_arithmetic_and_rejects_code(self) -> None:
        self.assertEqual(calculate("(12 + 8) * 2.5"), "50.0")
        with self.assertRaises(ValueError):
            calculate("__import__('os').system('echo unsafe')")
        with self.assertRaises(ValueError):
            calculate("1 / 0")

    def test_agent_executes_tool_calls_and_returns_status(self) -> None:
        messages = [{"role": "user", "content": "Add one to four"}]
        assistant_tool_call = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "add_one",
                        "arguments": {"value": 4},
                    }
                }
            ],
        }
        assistant_final = {"role": "assistant", "content": "5", "tool_calls": []}
        with (
            patch("libs.ollama_engine.chat", return_value=object()),
            patch(
                "libs.ollama_engine.print_streamed",
                side_effect=[(assistant_tool_call, 2), (assistant_final, 3)],
            ),
        ):
            outcome = run_agent_turn(
                messages, {"add_one": lambda value: value + 1}
            )
        self.assertEqual(outcome["status"], "success")
        self.assertEqual(outcome["content"], "5")
        self.assertEqual(outcome["tokens"], 5)
        self.assertEqual(messages[-2], {"role": "tool", "content": "5"})

    def test_agent_enforces_step_and_token_caps(self) -> None:
        tool_call = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "unused", "arguments": {}}}
            ],
        }
        with (
            patch("libs.ollama_engine.chat", return_value=object()),
            patch("libs.ollama_engine.print_streamed", return_value=(tool_call, 1)),
        ):
            step_limited = run_agent_turn(
                [{"role": "user", "content": "test"}],
                {"unused": lambda: "ok"},
                max_steps=1,
            )
        self.assertEqual(step_limited["status"], "max_steps")

        with (
            patch("libs.ollama_engine.chat", return_value=object()),
            patch("libs.ollama_engine.print_streamed", return_value=(tool_call, 2)),
        ):
            token_limited = run_agent_turn(
                [{"role": "user", "content": "test"}],
                {"unused": lambda: "ok"},
                max_tokens=2,
            )
        self.assertEqual(token_limited["status"], "token_limit")

    def test_evaluator_writes_three_trial_logs_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "run.jsonl"
            with patch("evaluate._expected_value", return_value=42):
                summary = evaluate.run_evaluation(
                    log_path=log_path,
                    agent_turn=lambda _messages: {
                        "content": "42",
                        "status": "success",
                        "tokens": 7,
                        "error": None,
                    },
                )
            self.assertEqual(len(summary["results"]), 16)
            self.assertEqual(summary["pass_at_1"], 1.0)
            self.assertEqual(summary["pass^3"], 1.0)
            with log_path.open(encoding="utf-8") as log_file:
                records = [json.loads(line) for line in log_file]
            self.assertEqual(sum(row["type"] == "trial" for row in records), 48)
            self.assertEqual(records[-1]["type"], "summary")

    def test_all_benchmark_reference_queries_are_valid(self) -> None:
        expected = [
            evaluate._expected_value(question["sql"])
            for question in evaluate.QUESTIONS
        ]
        self.assertEqual(len(expected), 16)
        self.assertTrue(all(value is not None for value in expected))


if __name__ == "__main__":
    unittest.main()
