# LangGraph - Security

## Input Guardrails

Validate and sanitize all user inputs before they reach the LLM:

```python
async def input_guard(state: State) -> dict:
    user_msg = state["messages"][-1].content

    # Prompt injection detection
    injection_patterns = ["ignore previous", "system:", "you are now", "disregard"]
    if any(p in user_msg.lower() for p in injection_patterns):
        return {"messages": [AIMessage(content="I can't process that request.")], "blocked": True}

    # Topic relevance check
    check = await llm.ainvoke(f"Is this about healthcare/billing? Answer YES/NO: {user_msg}")
    if "NO" in check.content:
        return {"messages": [AIMessage(content="I can only help with care-related topics.")], "blocked": True}

    return {"blocked": False}
```

## Output Guardrails

Redact sensitive data from LLM responses:

```python
import re

def redact_pii(text: str) -> str:
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REDACTED]', text)
    text = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD REDACTED]', text)
    text = re.sub(r'[\w.-]+@[\w.-]+\.\w+', '[EMAIL REDACTED]', text)
    return text
```

## Rate Limiting

Per-user throttling to prevent abuse:

```python
from collections import defaultdict
import time

rate_limits = defaultdict(list)  # user_id -> [timestamps]

def check_rate_limit(user_id: str, max_calls=3, window_seconds=10) -> bool:
    now = time.time()
    rate_limits[user_id] = [t for t in rate_limits[user_id] if now - t < window_seconds]
    if len(rate_limits[user_id]) >= max_calls:
        return False
    rate_limits[user_id].append(now)
    return True
```

## Sensitive State Handling

Don't persist secrets in checkpoints:

```python
async def cleanup_node(state: State) -> dict:
    return {"api_key": None, "auth_token": None}  # clear before checkpoint
```

## Grounding / Hallucination Check

Verify LLM output against source context:

```python
async def grounding_check(state: State) -> dict:
    check_prompt = f"Is this response grounded in the context? Context: {state['context']} Response: {state['response']}"
    result = await llm.ainvoke(check_prompt)
    if "not grounded" in result.content.lower():
        return {"needs_retry": True}
    return {"needs_retry": False}
```

## Security Checklist

- [ ] Input validation on all user-facing nodes
- [ ] PII redaction on all outputs
- [ ] Rate limiting per user/session
- [ ] API keys in env vars, never in state
- [ ] Checkpoint TTL cleanup for old sessions
- [ ] Tool permissions scoped per agent role
- [ ] Audit logging for all LLM calls
