"""
Streaming Research Agent — with LangSmith tracing
---------------------------------------------------
Every step is traced:
  - Top-level "Research Agent" chain run per query
  - DuckDuckGo web search (child run)
  - ChromaDB RAG retrieval (child run)
  - Gemini 2.5 Flash generation (child LLM run, with token usage + latency)
"""

import asyncio
import os
import threading
import time
from typing import AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langsmith import traceable
from langsmith.run_trees import RunTree

from mcp_tools import web_search_structured
from rag import RAGPipeline

load_dotenv()

MODEL = "gemini-2.5-flash"
MAX_RAG_CHUNKS = 3
MAX_WEB_RESULTS = 6
_TRACING = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
_LS_PROJECT = os.getenv("LANGCHAIN_PROJECT", "ai-research-assistant")

SYSTEM_PROMPT = """\
You are an expert AI Research Assistant with access to live web search results.

Answer the user's question using the web search results provided as your primary source.
Additional stored context from the knowledge base may also be included.

Rules:
- Base your answer primarily on the web search results provided.
- Be accurate, detailed, and well-structured.
- Use Markdown: headers (##), bold, bullet lists, code blocks where appropriate.
- Cite sources inline as [Source 1], [Source 2], etc. referencing the numbered web results.
- If the search results directly answer the question, use them — do not say you cannot access the web.
- Be comprehensive. The user wants a real, informative answer.
"""


# ── LangSmith RunTree helpers ────────────────────────────────────────────────

def _start_run(name: str, run_type: str, inputs: dict,
               parent: Optional[RunTree] = None) -> Optional[RunTree]:
    if not _TRACING:
        return None
    try:
        kwargs = dict(name=name, run_type=run_type, inputs=inputs,
                      project_name=_LS_PROJECT)
        run = parent.create_child(**kwargs) if parent else RunTree(**kwargs)
        run.post()
        return run
    except Exception:
        return None


def _end_run(run: Optional[RunTree], outputs: dict, error: str = None) -> None:
    if run is None:
        return
    try:
        if error:
            run.end(error=error)
        else:
            run.end(outputs=outputs)
        run.patch()
    except Exception:
        pass


# ── Agent ────────────────────────────────────────────────────────────────────

class ResearchAgent:
    def __init__(self):
        self.rag = RAGPipeline()
        self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    # ── Traced sub-operations (children of the root run) ─────────────────

    @traceable(run_type="tool", name="Web Search", tags=["web-search"])
    async def _search_web(self, query: str) -> tuple:
        """Traced async wrapper — LangSmith records latency + outputs."""
        return await asyncio.to_thread(web_search_structured, query, MAX_WEB_RESULTS)

    @traceable(run_type="retriever", name="RAG Retrieval", tags=["chromadb", "rag"])
    async def _search_rag(self, query: str) -> list:
        """Traced async wrapper — LangSmith records score + chunks returned."""
        return await asyncio.to_thread(self.rag.search, query, MAX_RAG_CHUNKS)

    # ── Main streaming loop ───────────────────────────────────────────────

    async def stream(self, message: str) -> AsyncGenerator[Dict, None]:
        sources: List[Dict] = []
        context_parts: List[str] = []

        # Top-level trace for the entire request
        root = _start_run(
            name="Research Agent",
            run_type="chain",
            inputs={"question": message},
        )

        try:
            # ── Step 1: Live web search ───────────────────────────────────
            yield {"type": "tool_call", "tool": "web_search", "query": message}
            await asyncio.sleep(0)

            web_text = ""
            try:
                search_text, results = await self._search_web(message)

                if results:
                    web_text = search_text
                    context_parts.append(f"=== WEB SEARCH RESULTS ===\n{search_text}")
                    for r in results:
                        sources.append({
                            "type": "web_search", "icon": "🌐",
                            "label": r["title"] or r["url"],
                            "detail": r["url"], "url": r["url"],
                        })
                    yield {"type": "tool_result",
                           "content": f"Found {len(results)} results"}

                    asyncio.create_task(
                        asyncio.to_thread(
                            self.rag.add_text, search_text, f"web:{message[:80]}"
                        )
                    )
                else:
                    yield {"type": "status",
                           "content": "Web search returned no results — using Gemini knowledge"}

            except Exception as exc:
                yield {"type": "status",
                       "content": f"Web search error: {exc} — using Gemini knowledge"}

            # ── Step 2: RAG supplementary context ────────────────────────
            try:
                rag_chunks = await self._search_rag(message)
                good = [
                    c for c in rag_chunks
                    if c.metadata.get("score", 0) >= 0.6
                    and not c.metadata.get("source", "").startswith(
                        f"web:{message[:20]}"
                    )
                ]
                if good:
                    context_parts.append(
                        f"=== KNOWLEDGE BASE ===\n{self.rag.build_context(good)}"
                    )
                    for chunk in good[:2]:
                        sources.append({
                            "type": "knowledge_base", "icon": "📚",
                            "label": chunk.metadata.get("source", "Knowledge Base"),
                            "detail": f"relevance: {chunk.metadata.get('score', '')}",
                        })
            except Exception:
                pass

            # ── Step 3: Stream Gemini answer ──────────────────────────────
            yield {"type": "status", "content": "Generating answer…"}
            await asyncio.sleep(0)

            full_context = "\n\n".join(context_parts)
            user_content = (
                f"Question: {message}\n\n{full_context}"
                if full_context else f"Question: {message}"
            )

            try:
                async for token in self._stream_tokens(user_content, parent=root):
                    yield {"type": "token", "content": token}
            except Exception as exc:
                yield {"type": "error", "content": f"Generation error: {exc}"}
                _end_run(root, {}, error=str(exc))
                return

            sources.append({
                "type": "gemini", "icon": "✨",
                "label": "Gemini 2.5 Flash", "detail": "Google DeepMind",
            })
            yield {"type": "sources", "sources": sources}
            yield {"type": "done"}

            _end_run(root, {
                "sources_used": len(sources),
                "web_results": len([s for s in sources if s["type"] == "web_search"]),
                "rag_chunks": len([s for s in sources if s["type"] == "knowledge_base"]),
            })

        except Exception as exc:
            _end_run(root, {}, error=str(exc))
            raise

    # ── Pipeline visualization stream ─────────────────────────────────────

    async def pipeline_stream(self, message: str) -> AsyncGenerator[Dict, None]:
        """
        Like stream() but emits granular pipeline_step events so the /pipeline
        visualization page can animate each stage with live timing & data.
        """
        t0 = time.time()

        def ms() -> int:
            return round((time.time() - t0) * 1000)

        def step(step_id: str, status: str, data: dict = None):
            return {
                "type": "pipeline_step",
                "step": step_id,
                "status": status,
                "data": data or {},
                "elapsed_ms": ms(),
            }

        # ① Query received
        yield step("query", "complete", {"text": message[:120]})
        await asyncio.sleep(0)

        # ② FastAPI router
        yield step("router", "complete", {"endpoint": "POST /pipeline/stream", "latency_ms": ms()})
        await asyncio.sleep(0)

        # ③ Web search
        yield step("web_search", "active", {"query": message[:80]})
        await asyncio.sleep(0)

        ws_t = time.time()
        search_text, results = "", []
        try:
            search_text, results = await asyncio.to_thread(
                web_search_structured, message, MAX_WEB_RESULTS
            )
            ws_ms = round((time.time() - ws_t) * 1000)
            yield step("web_search", "complete", {
                "results": len(results),
                "latency_ms": ws_ms,
                "top_result": results[0]["title"][:55] if results else "No results",
            })
            if search_text:
                asyncio.create_task(
                    asyncio.to_thread(self.rag.add_text, search_text, f"web:{message[:80]}")
                )
        except Exception as exc:
            yield step("web_search", "error", {"error": str(exc)[:80]})

        # ④ RAG retrieval (runs after web search for simplicity)
        yield step("rag", "active", {"query": message[:80], "kb_size": self.rag.total_chunks()})
        await asyncio.sleep(0)

        rag_t = time.time()
        rag_chunks, good_chunks = [], []
        try:
            rag_chunks = await asyncio.to_thread(self.rag.search, message, MAX_RAG_CHUNKS)
            rag_ms = round((time.time() - rag_t) * 1000)
            good_chunks = [
                c for c in rag_chunks
                if c.metadata.get("score", 0) >= 0.6
                and not c.metadata.get("source", "").startswith(f"web:{message[:20]}")
            ]
            yield step("rag", "complete", {
                "chunks_found": len(rag_chunks),
                "chunks_used": len(good_chunks),
                "latency_ms": rag_ms,
                "kb_total": self.rag.total_chunks(),
            })
        except Exception as exc:
            yield step("rag", "error", {"error": str(exc)[:80]})

        # ⑤ Context assembly
        yield step("context", "active", {})
        await asyncio.sleep(0)

        context_parts = []
        if results:
            context_parts.append(f"=== WEB SEARCH RESULTS ===\n{search_text}")
        if good_chunks:
            context_parts.append(f"=== KNOWLEDGE BASE ===\n{self.rag.build_context(good_chunks)}")

        full_context = "\n\n".join(context_parts)
        user_content = (
            f"Question: {message}\n\n{full_context}" if full_context else f"Question: {message}"
        )
        yield step("context", "complete", {
            "web_chars": len(search_text),
            "rag_chunks": len(good_chunks),
            "total_chars": len(user_content),
        })
        await asyncio.sleep(0)

        # ⑥ Gemini generation
        yield step("gemini", "active", {"model": MODEL, "input_chars": len(user_content)})
        await asyncio.sleep(0)

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        meta = {"prompt_tokens": 0, "completion_tokens": 0, "text": []}
        gen_t = time.time()
        token_count = 0

        def _pworker():
            try:
                for chunk in self._client.models.generate_content_stream(
                    model=MODEL,
                    contents=user_content,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT, temperature=1.0
                    ),
                ):
                    um = getattr(chunk, "usage_metadata", None)
                    if um:
                        meta["prompt_tokens"] = getattr(um, "prompt_token_count", 0) or 0
                        meta["completion_tokens"] = getattr(um, "candidates_token_count", 0) or 0
                    try:
                        text = chunk.text
                    except Exception:
                        text = None
                    if text:
                        meta["text"].append(text)
                        loop.call_soon_threadsafe(queue.put_nowait, text)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, f"*[Error: {exc}]*")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=_pworker, daemon=True).start()

        # ⑦ SSE streaming tokens
        yield step("sse", "active", {"status": "streaming"})
        await asyncio.sleep(0)

        while True:
            token = await queue.get()
            if token is None:
                break
            token_count += 1
            yield {"type": "token", "content": token}
            if token_count % 20 == 0:
                tps = round(token_count / max(0.001, time.time() - gen_t))
                yield step("sse", "active", {"tokens": token_count, "tokens_per_sec": tps})

        gen_ms = round((time.time() - gen_t) * 1000)
        total_ms = ms()

        yield step("gemini", "complete", {
            "prompt_tokens": meta["prompt_tokens"],
            "completion_tokens": meta["completion_tokens"],
            "total_tokens": meta["prompt_tokens"] + meta["completion_tokens"],
            "latency_ms": gen_ms,
        })
        yield step("sse", "complete", {"tokens_streamed": token_count, "latency_ms": gen_ms})

        sources = [
            {"type": "web_search", "icon": "🌐", "label": r["title"] or r["url"],
             "detail": r["url"], "url": r["url"]} for r in results
        ]
        if good_chunks:
            for c in good_chunks[:2]:
                sources.append({"type": "knowledge_base", "icon": "📚",
                                "label": c.metadata.get("source", "KB"), "detail": ""})
        sources.append({"type": "gemini", "icon": "✨", "label": "Gemini 2.5 Flash",
                        "detail": "Google DeepMind"})

        yield {"type": "sources", "sources": sources}
        yield step("complete", "complete", {
            "total_ms": total_ms,
            "web_results": len(results),
            "rag_chunks": len(good_chunks),
            "tokens": token_count,
        })
        yield {"type": "done"}

    # ── Gemini streaming bridge with LangSmith LLM run ───────────────────

    async def _stream_tokens(self, content: str,
                             parent: Optional[RunTree] = None) -> AsyncGenerator[str, None]:
        """
        Runs Gemini streaming in a daemon thread, bridges tokens to async via Queue.
        After all tokens arrive, logs a full LLM run to LangSmith with:
          - prompt / completion text
          - prompt_tokens / completion_tokens / total_tokens
          - latency in seconds
        """
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        meta = {"prompt_tokens": 0, "completion_tokens": 0, "text": []}
        t0 = time.time()

        def _worker():
            try:
                for chunk in self._client.models.generate_content_stream(
                    model=MODEL,
                    contents=content,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        temperature=1.0,
                    ),
                ):
                    um = getattr(chunk, "usage_metadata", None)
                    if um:
                        meta["prompt_tokens"] = getattr(um, "prompt_token_count", 0) or 0
                        meta["completion_tokens"] = (
                            getattr(um, "candidates_token_count", 0) or 0
                        )
                    try:
                        text = chunk.text
                    except Exception:
                        text = None
                    if text:
                        meta["text"].append(text)
                        loop.call_soon_threadsafe(queue.put_nowait, text)
            except Exception as exc:
                err = f"\n\n*[Stream error: {exc}]*"
                meta["text"].append(err)
                loop.call_soon_threadsafe(queue.put_nowait, err)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=_worker, daemon=True).start()

        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

        # ── Log completed LLM run to LangSmith ───────────────────────────
        latency = round(time.time() - t0, 3)
        total = meta["prompt_tokens"] + meta["completion_tokens"]
        llm_run = _start_run(
            name="Gemini 2.5 Flash",
            run_type="llm",
            inputs={
                "prompts": [content[:3000]],
                "model": MODEL,
                "temperature": 1.0,
            },
            parent=parent,
        )
        _end_run(llm_run, {
            "generations": [{"text": "".join(meta["text"])}],
            "llm_output": {
                "model": MODEL,
                "token_usage": {
                    "prompt_tokens": meta["prompt_tokens"],
                    "completion_tokens": meta["completion_tokens"],
                    "total_tokens": total,
                },
            },
            "latency_seconds": latency,
        })
