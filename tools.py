"""Tool definitions and the fake Acme Cloud backend.

Each tool has two halves:
  1. A schema Claude sees (TOOLS). The description is prompt engineering:
     Claude decides when and how to call a tool mostly from this text.
  2. A Python function that runs when Claude calls it (Backend methods).
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# `strict: True` guarantees the arguments Claude sends match the schema exactly.
# Strict schemas need `additionalProperties: False` and every property in `required`.
TOOLS = [
    {
        "name": "lookup_account",
        "description": (
            "Look up a customer's account by email address. Returns the plan, "
            "status and recent invoices. Call this before any billing action."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "The customer's email address."}
            },
            "required": ["email"],
            "additionalProperties": False,
        },
    },
    {
        "name": "search_kb",
        "description": (
            "Search the Acme Cloud knowledge base. Returns up to three articles. "
            "Use it to answer policy and how-to questions; do not answer those from memory."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A few keywords, e.g. 'refund policy'."}
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "issue_refund",
        "description": (
            "Refund money to a customer for a specific invoice. Only use it when the "
            "refund policy in the knowledge base allows the refund."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {"type": "string"},
                "invoice_id": {"type": "string"},
                "amount_usd": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["account_id", "invoice_id", "amount_usd", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "escalate_to_human",
        "description": (
            "Hand the ticket to a human specialist. Use it when policy requires a human "
            "or the knowledge base does not cover the question."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "team": {"type": "string", "enum": ["billing", "security", "legal", "general"]},
                "summary": {"type": "string", "description": "What the human needs to know."},
            },
            "required": ["team", "summary"],
            "additionalProperties": False,
        },
    },
]


class ToolError(Exception):
    """Raise this from a tool to send Claude an error result it can react to."""


class Backend:
    """In-memory fake of Acme's systems. Create a fresh one per ticket so runs don't leak state."""

    def __init__(self):
        self.accounts = json.loads((DATA_DIR / "accounts.json").read_text())
        self.kb = json.loads((DATA_DIR / "kb.json").read_text())
        self.refunds = []       # every refund issued during this ticket
        self.escalations = []   # every escalation raised during this ticket

    def lookup_account(self, email: str) -> dict:
        for account in self.accounts:
            if account["email"].lower() == email.strip().lower():
                return account
        raise ToolError(f"No account found for {email}. Ask the customer to confirm their email.")

    def search_kb(self, query: str) -> list[dict]:
        words = set(query.lower().split())
        scored = []
        for article in self.kb:
            text = (article["title"] + " " + article["body"]).lower()
            score = sum(1 for word in words if word in text)
            if score:
                scored.append((score, article))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [article for _, article in scored[:3]]

    def issue_refund(self, account_id: str, invoice_id: str, amount_usd: float, reason: str) -> dict:
        account = next((a for a in self.accounts if a["account_id"] == account_id), None)
        if account is None:
            raise ToolError(f"Unknown account_id {account_id}.")
        invoice = next((i for i in account["invoices"] if i["invoice_id"] == invoice_id), None)
        if invoice is None:
            raise ToolError(f"Invoice {invoice_id} does not belong to account {account_id}.")
        if amount_usd <= 0 or amount_usd > invoice["amount_usd"]:
            raise ToolError(f"Refund must be between $0 and the invoice amount (${invoice['amount_usd']}).")

        # TODO (milestone 2, guardrail): refunds over $100 need human approval.
        # Enforce it HERE in code, not only in the prompt, by raising ToolError with a
        # message that tells Claude what to do instead (escalate to the billing team).
        # Then check that eval case t03 passes.
        if amount_usd > 100:
            raise ToolError("Refunds over $100 need human approval. Please escalate to the billing team.")

        refund = {"account_id": account_id, "invoice_id": invoice_id, "amount_usd": amount_usd, "reason": reason}
        self.refunds.append(refund)
        return {"status": "refunded", **refund}

    def escalate_to_human(self, team: str, summary: str) -> dict:
        ticket = {"team": team, "summary": summary}
        self.escalations.append(ticket)
        return {"status": "escalated", "queue": team}


def execute_tool(backend: Backend, name: str, tool_input: dict) -> tuple[str, bool]:
    """Run one tool call. Returns (result_text, is_error).

    Errors go back to Claude as a tool_result with is_error=True rather than
    crashing the loop, so the model can recover (retry, ask the customer, escalate).
    """
    handler = getattr(backend, name, None)
    if handler is None or name.startswith("_"):
        return f"Unknown tool: {name}", True
    try:
        return json.dumps(handler(**tool_input)), False
    except ToolError as error:
        return str(error), True
