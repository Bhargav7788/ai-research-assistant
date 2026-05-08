"""
Streaming Research Agent
------------------------
Flow for EVERY query:
  1. Live web search with the exact user question (always first)
  2. RAG/ChromaDB checked for additional stored context
  3. Both combined → Gemini 2.5 Flash streams the answer
  4. Web sources shown at the bottom of every response
"""

import asyncio
import os
import threading
from typing import AsyncGenerator, Dict, List

from dotenv import load_dotenv
from google import genai
from google.genai import types

from mcp_tools import web_search_structured
from rag import RAGPipeline

load_dotenv()

MODEL = "gemini-2.5-flash"
MAX_RAG_CHUNKS = 3
MAX_WEB_RESULTS = 6

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


class ResearchAgent:
    def __init__(self):
        self.rag = RAGPipeline()
        self._client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    async def stream(self, message: str) -> AsyncGenerator[Dict, None]:
        sources: List[Dict] = []
        context_parts: List[str] = []

        # ── Step 1: Live web search — ALWAYS, with the exact question ──
        yield {"type": "tool_call", "tool": "web_search", "query": message}
        await asyncio.sleep(0)

        web_text = ""
        try:
            search_text, results = await asyncio.to_thread(
                web_search_structured, message, MAX_WEB_RESULTS
            )

            if results:
                web_text = search_text
                context_parts.append(f"=== WEB SEARCH RESULTS ===\n{search_text}")

                for r in results:
                    sources.append({
                        "type": "web_search",
                        "icon": "🌐",
                        "label": r["title"] or r["url"],
                        "detail": r["url"],
                        "url": r["url"],
                    })

                yield {"type": "tool_result", "content": f"Found {len(results)} results"}

                # Save to RAG in background for future recall
                asyncio.create_task(
                    asyncio.to_thread(self.rag.add_text, search_text, f"web:{message[:80]}")
                )
            else:
                yield {"type": "status", "content": "Web search returned no results — using Gemini knowledge"}

        except Exception as exc:
            yield {"type": "status", "content": f"Web search error: {exc} — using Gemini knowledge"}

        # ── Step 2: RAG — supplementary context if anything stored ─────
        try:
            rag_chunks = await asyncio.to_thread(self.rag.search, message, MAX_RAG_CHUNKS)
            # Only use RAG chunks that weren't just added from this web search
            good_chunks = [
                c for c in rag_chunks
                if c.metadata.get("score", 0) >= 0.6
                and not c.metadata.get("source", "").startswith(f"web:{message[:20]}")
            ]
            if good_chunks:
                rag_context = self.rag.build_context(good_chunks)
                context_parts.append(f"=== KNOWLEDGE BASE (previously stored) ===\n{rag_context}")
                for chunk in good_chunks[:2]:
                    sources.append({
                        "type": "knowledge_base",
                        "icon": "📚",
                        "label": chunk.metadata.get("source", "Knowledge Base"),
                        "detail": f"relevance: {chunk.metadata.get('score', '')}",
                    })
        except Exception:
            pass  # RAG is supplementary — never block the answer

        # ── Step 3: Stream Gemini answer ────────────────────────────────
        yield {"type": "status", "content": "Generating answer…"}
        await asyncio.sleep(0)

        full_context = "\n\n".join(context_parts)
        user_content = f"Question: {message}\n\n{full_context}" if full_context else f"Question: {message}"

        try:
            async for token in self._stream_tokens(user_content):
                yield {"type": "token", "content": token}
        except Exception as exc:
            yield {"type": "error", "content": f"Generation error: {exc}"}
            return

        sources.append({"type": "gemini", "icon": "✨", "label": "Gemini 2.5 Flash", "detail": "Google DeepMind"})
        yield {"type": "sources", "sources": sources}
        yield {"type": "done"}

    # ── Sync → async bridge for Gemini token streaming ─────────────────

    async def _stream_tokens(self, content: str) -> AsyncGenerator[str, None]:
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

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
                    try:
                        text = chunk.text
                    except Exception:
                        text = None
                    if text:
                        loop.call_soon_threadsafe(queue.put_nowait, text)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, f"\n\n*[Stream error: {exc}]*")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=_worker, daemon=True).start()

        while True:
            token = await queue.get()
            if token is None:
                break
            yield token
