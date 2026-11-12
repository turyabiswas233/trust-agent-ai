import argparse
import json
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

from libs.model_config import SYSTEM_PROMPT
from libs.ollama_engine import run_agent_turn
from tools.sql_tools import execute_sql

TRIALS = 3
QUESTIONS = [
    {
        "id": "customer-count",
        "prompt": "How many customers are in the database?",
        "sql": "SELECT COUNT(*) FROM customers",
    },
    {
        "id": "product-count",
        "prompt": "How many products are listed?",
        "sql": "SELECT COUNT(*) FROM products",
    },
    {
        "id": "order-count",
        "prompt": "How many orders have been placed?",
        "sql": "SELECT COUNT(*) FROM orders",
    },
    {
        "id": "city-count",
        "prompt": "How many distinct customer cities are represented?",
        "sql": "SELECT COUNT(DISTINCT city) FROM customers",
    },
    {
        "id": "total-revenue",
        "prompt": "What is the total sales revenue across all order items?",
        "sql": "SELECT ROUND(SUM(quantity * unit_price), 2) FROM order_items",
    },
    {
        "id": "average-product-price",
        "prompt": "What is the average listed product price?",
        "sql": "SELECT ROUND(AVG(price), 2) FROM products",
    },
    {
        "id": "most-common-status",
        "prompt": "Which order status occurs most often?",
        "sql": (
            "SELECT status FROM orders GROUP BY status "
            "ORDER BY COUNT(*) DESC, status LIMIT 1"
        ),
    },
    {
        "id": "most-ordered-product",
        "prompt": "Which product has the highest total quantity sold?",
        "sql": (
            "SELECT p.name FROM order_items oi JOIN products p "
            "ON p.id = oi.product_id GROUP BY p.id "
            "ORDER BY SUM(oi.quantity) DESC, p.name LIMIT 1"
        ),
    },
    {
        "id": "highest-spending-customer",
        "prompt": "Which customer has spent the most across all order items?",
        "sql": (
            "SELECT c.name FROM customers c JOIN orders o ON o.customer_id = c.id "
            "JOIN order_items oi ON oi.order_id = o.id "
            "GROUP BY c.id ORDER BY SUM(oi.quantity * oi.unit_price) DESC, c.name LIMIT 1"
        ),
    },
    {
        "id": "delivered-orders",
        "prompt": "How many orders have been delivered?",
        "sql": "SELECT COUNT(*) FROM orders WHERE status = 'delivered'",
    },
    {
        "id": "average-order-value",
        "prompt": "What is the average order value, counting each item's quantity?",
        "sql": (
            "SELECT ROUND(AVG(order_total), 2) FROM "
            "(SELECT order_id, SUM(quantity * unit_price) AS order_total "
            "FROM order_items GROUP BY order_id)"
        ),
    },
    {
        "id": "most-expensive-product",
        "prompt": "What is the name of the most expensive product?",
        "sql": "SELECT name FROM products ORDER BY price DESC, name LIMIT 1",
    },
    {
        "id": "repeat-customers",
        "prompt": "How many customers have placed more than five orders?",
        "sql": (
            "SELECT COUNT(*) FROM (SELECT customer_id FROM orders "
            "GROUP BY customer_id HAVING COUNT(*) > 5)"
        ),
    },
    {
        "id": "busiest-month",
        "prompt": "Which calendar month (YYYY-MM) had the most orders?",
        "sql": (
            "SELECT strftime('%Y-%m', order_date) AS month FROM orders "
            "GROUP BY month ORDER BY COUNT(*) DESC, month LIMIT 1"
        ),
    },
    {
        "id": "top-revenue-category",
        "prompt": "Which product category generated the most sales revenue?",
        "sql": (
            "SELECT p.category FROM order_items oi JOIN products p "
            "ON p.id = oi.product_id GROUP BY p.category "
            "ORDER BY SUM(oi.quantity * oi.unit_price) DESC, p.category LIMIT 1"
        ),
    },
    {
        "id": "units-sold",
        "prompt": "How many total units have been sold across all order items?",
        "sql": "SELECT SUM(quantity) FROM order_items",
    },
]


def _expected_value(reference_sql: str) -> Any:
    result = json.loads(execute_sql(reference_sql))
    if result["row_count"] != 1 or len(result["rows"][0]) != 1:
        raise ValueError(f"Benchmark reference query must return one value: {reference_sql}")
    return result["rows"][0][0]


def _answer_matches(answer: str, expected: Any) -> bool:
    if expected is None:
        return "null" in answer.casefold() or "no data" in answer.casefold()
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            target = Decimal(str(expected))
        except InvalidOperation:
            return False
        candidates = re.findall(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?(?![\w.])", answer)
        for candidate in candidates:
            try:
                if abs(Decimal(candidate.replace(",", "")) - target) <= Decimal("0.01"):
                    return True
            except InvalidOperation:
                continue
        return False
    return str(expected).casefold() in answer.casefold()


def run_evaluation(
    trials: int = TRIALS,
    log_path: str | Path | None = None,
    agent_turn: Callable[..., dict[str, Any]] = run_agent_turn,
) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("trials must be at least one.")
    if log_path is None:
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        log_path = Path("output/evaluations") / f"evaluation-{stamp}.jsonl"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    question_results = []
    with log_path.open("w", encoding="utf-8") as log_file:
        for question in QUESTIONS:
            expected = _expected_value(question["sql"])
            trial_results = []
            for trial in range(1, trials + 1):
                conversation = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question["prompt"]},
                ]
                outcome = agent_turn(conversation)
                answer = str(outcome.get("content") or "")
                passed = (
                    outcome.get("status") == "success"
                    and _answer_matches(answer, expected)
                )
                record = {
                    "type": "trial",
                    "question_id": question["id"],
                    "trial": trial,
                    "prompt": question["prompt"],
                    "expected": expected,
                    "answer": answer,
                    "passed": passed,
                    "status": outcome.get("status"),
                    "tokens": outcome.get("tokens", 0),
                    "error": outcome.get("error"),
                }
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()
                trial_results.append(record)
            question_results.append(
                {
                    "question_id": question["id"],
                    "expected": expected,
                    "trials": trial_results,
                    "pass_at_1": trial_results[0]["passed"],
                    "pass_all_trials": all(item["passed"] for item in trial_results),
                }
            )

        summary = {
            "questions": len(QUESTIONS),
            "trials_per_question": trials,
            "pass_at_1": sum(item["pass_at_1"] for item in question_results)
            / len(question_results),
            "pass^3": (
                sum(item["pass_all_trials"] for item in question_results)
                / len(question_results)
                if trials == 3
                else None
            ),
            "log_path": str(log_path),
            "results": question_results,
        }
        log_file.write(json.dumps({"type": "summary", **summary}, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the SQLite agent on 16 questions.")
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    summary = run_evaluation(trials=args.trials, log_path=args.log)
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
