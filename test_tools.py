"""Offline checks for the tools and the grader. No API key needed.

    python test_tools.py
"""

import json

from tools import TOOLS, Backend, execute_tool


def test_lookup_account():
    text, is_error = execute_tool(Backend(), "lookup_account", {"email": "DANA@brightloop.io"})
    assert not is_error and json.loads(text)["account_id"] == "A-1001"


def test_lookup_unknown_email_is_a_tool_error():
    text, is_error = execute_tool(Backend(), "lookup_account", {"email": "nobody@example.com"})
    assert is_error and "No account found" in text


def test_search_kb_finds_refund_policy():
    text, _ = execute_tool(Backend(), "search_kb", {"query": "refund policy"})
    assert json.loads(text)[0]["id"] == "KB-01"


def test_refund_is_recorded():
    backend = Backend()
    args = {"account_id": "A-1001", "invoice_id": "INV-7702", "amount_usd": 49, "reason": "duplicate"}
    _, is_error = execute_tool(backend, "issue_refund", args)
    assert not is_error and backend.refunds[0]["amount_usd"] == 49


def test_refund_cannot_exceed_invoice():
    args = {"account_id": "A-1001", "invoice_id": "INV-7702", "amount_usd": 5000, "reason": "x"}
    _, is_error = execute_tool(Backend(), "issue_refund", args)
    assert is_error


def test_every_tool_schema_is_strict_and_has_a_handler():
    for tool in TOOLS:
        schema = tool["input_schema"]
        assert tool["strict"] is True and schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        assert hasattr(Backend, tool["name"])


# TODO (milestone 2): add a test that a $899 refund returns an error once your guardrail exists.
# in accounts.json, A-1002 has an invoice of $899
def test_refund_over_100_needs_approval():
    args = {"account_id": "A-1002", "invoice_id": "INV-8104", "amount_usd": 899, "reason": "x"}
    _, is_error = execute_tool(Backend(), "issue_refund", args)
    assert is_error

if __name__ == "__main__":
    tests = [value for name, value in dict(globals()).items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
