# LangGraph - Limitations

## Framework Limitations

1. **Steep learning curve** - Graph-based thinking is unfamiliar to most devs. Simple chatbots don't need it.

2. **State serialization overhead** - Full state is serialized/deserialized at every node. Large states (big message histories, large documents) cause performance hits.

3. **No native distributed execution** - All nodes run in a single process. For true distributed agents across machines, you need external orchestration (Celery, Temporal, etc.).

4. **Checkpointer lock-in** - Switching checkpointer backends requires migration. Schema changes between versions can break existing checkpoints.

5. **Debugging complexity** - Cyclic graphs are harder to reason about than linear chains. State mutations across nodes can produce surprising behavior.

6. **LangChain coupling** - While nodes are plain functions, LLM integration leans heavily on LangChain abstractions (ChatModel, Messages, Tools). Switching to raw API calls loses convenience features.

7. **Limited error recovery** - Retry policies are per-node. No built-in saga pattern or compensation logic for multi-node failures.

8. **Cold-start in serverless** - Graph compilation + LangChain imports add ~2-5s cold start overhead.

## When NOT to Use LangGraph

- Simple Q&A bots (use LangChain directly)
- Single LLM call with no branching
- Batch data pipelines (use Airflow/Prefect)
- Real-time streaming-only use cases (use WebSockets directly)

## Common Pitfalls

- **Unbounded message lists** - Without trimming, token costs and latency grow linearly with conversation length
- **Missing thread_id** - Forgetting configurable thread IDs causes state leaks between users
- **Sync in async** - Mixing sync/async nodes causes event loop blocking
- **Over-engineering** - Building a 10-node graph when 2 nodes would suffice
