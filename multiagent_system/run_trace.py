from datetime import datetime, timezone
from threading import Lock

from langchain_core.callbacks import BaseCallbackHandler


def jsonable(value):
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class RunTrace(BaseCallbackHandler):
    def __init__(self):
        self.events = []
        self._lock = Lock()

    def add(self, kind, **data):
        with self._lock:
            self.events.append({
                "sequence": len(self.events) + 1,
                "time": datetime.now(timezone.utc).isoformat(),
                "type": kind,
                **jsonable(data),
            })

    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
        self.add("model_input", run_id=run_id, parent_run_id=parent_run_id,
                 node=(kwargs.get("metadata") or {}).get("langgraph_node"),
                 model=serialized.get("id"), messages=messages)

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        self.add("model_output", run_id=run_id, parent_run_id=parent_run_id,
                 response=response)

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        self.add("model_error", run_id=run_id, parent_run_id=parent_run_id,
                 error=str(error))
