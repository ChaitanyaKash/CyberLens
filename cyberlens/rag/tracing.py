"""Optional Arize Phoenix tracing (OpenInference span conventions). A no-op unless CYBERLENS_TRACE=1 or
PHOENIX_COLLECTOR_ENDPOINT is set and arize-phoenix is installed. Only masked text is ever recorded."""
import contextlib
import json
import os

_tracer = None


def tracer():
    global _tracer
    if _tracer is None:
        _tracer = False
        if os.getenv("CYBERLENS_TRACE") or os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
            try:
                from phoenix.otel import register
                _tracer = register(project_name="cyberlens-rag", auto_instrument=False).get_tracer("cyberlens.rag")
            except ImportError:
                pass
    return _tracer


def _value(v):
    return v if isinstance(v, (str, int, float, bool)) else json.dumps(v, ensure_ascii=False, default=str)


@contextlib.contextmanager
def span(name, kind, **attributes):
    t = tracer()
    if not t:
        yield None
        return
    with t.start_as_current_span(name) as s:
        s.set_attribute("openinference.span.kind", kind)
        for key, value in attributes.items():
            s.set_attribute(key.replace("__", "."), _value(value))
        yield s


def record(s, **attributes):
    if s is not None:
        for key, value in attributes.items():
            s.set_attribute(key.replace("__", "."), _value(value))


def record_documents(s, hits):
    if s is None:
        return
    for i, h in enumerate(hits):
        prefix = f"retrieval.documents.{i}.document"
        s.set_attribute(f"{prefix}.id", h["chunk_id"])
        s.set_attribute(f"{prefix}.content", h["text"])
        s.set_attribute(f"{prefix}.score", float(h.get("score", 0)))
        s.set_attribute(f"{prefix}.metadata", _value({k: h.get(k) for k in ("id", "title", "section", "page")}))
