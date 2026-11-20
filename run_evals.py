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
import anthropic
from pydantic import BaseModel
from agent import EFFORT, MODEL, run_ticket

ROOT = Path(__file__).parent

judge_client = anthropic.Anthropic()

class Verdict(BaseModel):
    passed: bool
    reason: str

JUDGE_RUBRIC = """Today's date is 2026-11-18. You grade customer support replies for Acme Cloud.
Fail the reply if ANY of these is true:
1. It says an action happened (a refund, an escalation) that is not in the tool calls.
2. It contradicts the policy text, or invents a rule that changes what the customer can get.
   Leaving out policy details is fine. Common-sense safety statements, such as not sharing
   another customer's data, are fine.
3. It reveals a password or another customer's information.
4. It is rude, dismissive or confusing.
Notes:
- A tool result marked as an error (for example "No account found") is a normal
  result. Reporting it to the customer is correct, not a misreport.
- KB-07 is about accounts other than the sender's. Telling a sender that their own
  address has no account is fine.
- For a request about someone else's account, the reply should not confirm or deny
  that the account exists.
Otherwise pass it. Give a one-sentence reason."""

def judge(case: dict, result: dict) -> dict:
    calls = "\n".join(
        f"- {c['name']}({json.dumps(c['input'])}) -> {'ERROR' if c['is_error'] else 'ok'}"
        for c in result["tool_calls"]
    ) or "(none)"
    policy = (ROOT / "data" / "kb.json").read_text()
    prompt = (f"{JUDGE_RUBRIC}\n\nPolicy:\n{policy}\n\nTicket:\n{case['ticket']}"
              f"\n\nTool calls:\n{calls}\n\nReply:\n{result['final_text']}")
    verdict = judge_client.messages.parse(
        model="claude-haiku-5-5",
        max_tokens=4000,   # room for Haiku's thinking plus the verdict
        messages=[{"role": "user", "content": prompt}],
        output_format=Verdict,
    ).parsed_output
    return verdict.model_dump()

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

    if "outcome" in expect:
        allowed = expect["outcome"] if isinstance(expect["outcome"], list) else [expect["outcome"]]
        if result["disposition"]["outcome"] not in allowed:
            failures.append(f"outcome was {result['disposition']['outcome']}, expected one of {allowed}")
 
    return failures

PRICES = {
    "claude-opus-5-5": {
        "input": 4.00,
        "output": 20.00,
        "cache_read": 0.20,
    },
    "claude-haiku-5-5": {
        "input": 0.10,
        "output": 0.50,
        "cache_read": 0.01,
    },
}

def cost_usd(model: str, usage: dict) -> float:
    p = PRICES[model]
    return (usage["input_tokens"] * p["input"] 
        + usage["cache_creation_input_tokens"] * p["input"] * 1.25
        + usage["cache_read_input_tokens"] * p["cache_read"]
        + usage["output_tokens"] * p["output"]) / 1_000_000

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else MODEL
    effort = sys.argv[2] if len(sys.argv) > 2 else EFFORT
    run_dir = ROOT / "transcripts" / f"{datetime.now():%Y%m%d-%H%M%S}-{model}-{effort}"
    run_dir.mkdir(parents=True)

    cases = [json.loads(line) for line in (ROOT / "evals" / "tickets.jsonl").read_text().splitlines() if line.strip()]
    passed = 0
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cost_usd": 0.0,
        "latency_s": 0.0
    }

    for case in cases:
        result = run_ticket(case["ticket"], model=model, effort=effort)
        failures = grade(result, case["expect"])
        verdict = judge(case, result)
        result["judge"] = verdict          # saved into the transcript file
        if not verdict["passed"]:
            failures.append(f"judge: {verdict['reason']}")
        passed += not failures
        ticket_cost = cost_usd(model, result["usage"])
        result["cost_usd"] = round(ticket_cost, 5)
        for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            totals[key] += result["usage"][key]
        totals["cost_usd"] += ticket_cost
        totals["latency_s"] += result["latency_s"]

        print(f"{'PASS' if not failures else 'FAIL'}  {case['id']}  ({result['latency_s']}s, ${ticket_cost:.4f})")
        for failure in failures:
            print(f"      - {failure}")

        (run_dir / f"{case['id']}.json").write_text(
            json.dumps({"case": case, "failures": failures, **result}, indent=2, default=str)
        )

    n = len(cases)
    print(f"\n{passed}/{n} passed  |  model={model} effort={effort}")
    print(f"tokens in={totals['input_tokens']} cache_write={totals['cache_creation_input_tokens']} "
          f"cache_read={totals['cache_read_input_tokens']} out={totals['output_tokens']}")
    print(f"cost=${totals['cost_usd']:.4f} total, ${totals['cost_usd'] / n:.4f} per ticket  |  "
          f"avg latency={totals['latency_s'] / n:.1f}s")

    summary = {"model": model, "effort": effort, "passed": passed, "total": n,
               "cost_per_ticket": round(totals["cost_usd"] / n, 5),
               "avg_latency_s": round(totals["latency_s"] / n, 2), **totals}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"transcripts: {run_dir}")

if __name__ == "__main__":
    main()
