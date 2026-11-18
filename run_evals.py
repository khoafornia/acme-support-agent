"""Run every ticket in evals/tickets.jsonl through the agent and grade the results.

    python run_evals.py                      # default model and effort
    python run_evals.py claude-haiku-5-5     # another model
    python run_evals.py claude-opus-5-5 low  # another effort level

Each run saves full transcripts to transcripts/<run name>/ so you can read failures.
Every run calls the API and costs money: 5 tickets is cents, 25 tickets across
several models adds up, so check your usage in the Console.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from agent import EFFORT, MODEL, run_ticket

ROOT = Path(__file__).parent


def grade(result: dict, expect: dict) -> list[str]:
    """Code-based grader. Returns a list of failures; an empty list means pass."""
    failures = []
    called = {call["name"] for call in result["tool_calls"]}

    for name in expect.get("must_call", []):
        if name not in called:
            failures.append(f"expected a call to {name}")
    for name in expect.get("must_not_call", []):
        if name in called:
            failures.append(f"should not have called {name}")

    if "refund_total_usd" in expect:
        total = sum(refund["amount_usd"] for refund in result["refunds"])
        if total != expect["refund_total_usd"]:
            failures.append(f"refunded ${total}, expected ${expect['refund_total_usd']}")

    if result["stop_reason"] != "end_turn":
        failures.append(f"stopped with {result['stop_reason']}")

    # TODO (milestone 3): also grade the structured disposition once agent.py returns one.
    # TODO (milestone 4): add an LLM judge for things code can't check, such as tone or
    #   whether the reply states the policy correctly. Use a cheap model
    #   ("claude-haiku-5-5"), give it the ticket, the final reply and a short rubric,
    #   and ask for a structured verdict (pass/fail plus a one-line reason).
    return failures


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else MODEL
    effort = sys.argv[2] if len(sys.argv) > 2 else EFFORT
    run_dir = ROOT / "transcripts" / f"{datetime.now():%Y%m%d-%H%M%S}-{model}-{effort}"
    run_dir.mkdir(parents=True)

    cases = [json.loads(line) for line in (ROOT / "evals" / "tickets.jsonl").read_text().splitlines() if line.strip()]
    passed = 0
    totals = {"input_tokens": 0, "output_tokens": 0, "latency_s": 0.0}

    for case in cases:
        result = run_ticket(case["ticket"], model=model, effort=effort)
        failures = grade(result, case["expect"])
        passed += not failures
        totals["input_tokens"] += result["usage"]["input_tokens"]
        totals["output_tokens"] += result["usage"]["output_tokens"]
        totals["latency_s"] += result["latency_s"]

        print(f"{'PASS' if not failures else 'FAIL'}  {case['id']}  ({result['latency_s']}s)")
        for failure in failures:
            print(f"      - {failure}")

        (run_dir / f"{case['id']}.json").write_text(
            json.dumps({"case": case, "failures": failures, **result}, indent=2, default=str)
        )

    print(f"\n{passed}/{len(cases)} passed  |  model={model} effort={effort}")
    print(f"tokens in={totals['input_tokens']} out={totals['output_tokens']}  |  "
          f"avg latency={totals['latency_s'] / len(cases):.1f}s")
    print(f"transcripts: {run_dir}")
    # TODO (milestone 6): turn tokens into dollars per ticket using current prices,
    # and print a comparison table across models and effort levels.


if __name__ == "__main__":
    main()
