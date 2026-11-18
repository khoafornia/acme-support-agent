# Acme Cloud support agent

A small customer-support agent on the Claude API, with an eval suite. It is a practice
project for Applied AI interviews: build a pilot, measure it, recommend a setup.

This is starter code. The agent loop, the fake backend, four tools and a 5-case eval
runner work. The milestones below are left for you, marked `TODO` in the code.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate # alias = venv
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # create one at https://platform.claude.com
```

## Run it

```bash
python test_tools.py                       # offline check of the tools, no API key needed
python agent.py "Hi, this is dana@brightloop.io. I was charged twice on Nov 1."
python run_evals.py                        # all eval tickets, saves transcripts/
python run_evals.py claude-haiku-5-5 low   # another model and effort level
```

Expect `t03-refund-over-limit` to be the interesting one on a first run: nothing in the
code stops an $899 refund yet. Making that case pass reliably is milestone 2.

## Files

| File | What it is |
| --- | --- |
| `agent.py` | The tool loop, system prompt, model and effort settings |
| `tools.py` | Tool schemas Claude sees, and the fake backend that runs them |
| `run_evals.py` | Runs every ticket, grades with code, saves transcripts |
| `evals/tickets.jsonl` | Eval cases: a ticket plus the expected behavior |
| `data/` | Fake accounts and knowledge base |
| `test_tools.py` | Offline tests for the tools and the grader |

## Milestones

1. **Agent loop (done for you).** Read `agent.py` until you can explain each line: why the

   whole `response.content` is appended, why all tool results go back in one message, what
    each `stop_reason` means.
2. **Guardrail.** In `tools.py`, block refunds over $100 in code and return an error that

   tells Claude to escalate. Ask yourself why the prompt alone is not enough.
3. **Structured result.** End each ticket with a typed disposition using

   `client.messages.parse(..., output_format=PydanticModel)` and grade on it.
   Right now the only output is free text, which code can’t grade reliably.
   Structured outputs make Claude’s answer match a schema you define, so you can grade it with a simple `==`.
4. **Eval suite.** Grow `evals/tickets.jsonl` from 5 to 25 cases. Cover ambiguous requests,

   a wrong email, a customer asking for a password, a legal threat, two issues in one
    ticket, and more injection attempts. Add an LLM judge for tone and policy accuracy.
5. **Transcript analysis.** Read the failed transcripts, group failures by cause, change

   the prompt or the tool descriptions, rerun. Keep a before-and-after table.
6. **Cost and latency.** Add prompt caching and verify it with

   `usage.cache_read_input_tokens`. Compare `claude-opus-5-5` and `claude-haiku-5-5`, and
    two effort levels, on accuracy, cost per ticket and latency.

Stretch: serve the tools from an MCP server, run the evals through the Batch API (half
price, asynchronous), or add refusal fallbacks (the `fallbacks` parameter) so a declined
request is retried on another model.

## Write-up

Finish with one page: which model and settings you would recommend to Acme, and why, with
your numbers. That page is your interview story and the draft of a blog post.

## API details worth knowing (as of November 2026)

- `claude-opus-5-5` always thinks. You control depth with `output_config={"effort": ...}`;
its default is `medium`. Sending `thinking: {"type": "disabled"}` returns a 400.
- Forcing a tool with `tool_choice` `any` or `tool` returns a 400 on Opus 5.5. Use the
default `auto`, steer in the prompt, and keep `strict: True` on tools.
- A response can end with `stop_reason: "refusal"`. Check `stop_reason` before reading content.
- Caching is a prefix match: tools, then system, then messages. Any change early in the
prompt (a timestamp, reordered tools) silently disables it.
