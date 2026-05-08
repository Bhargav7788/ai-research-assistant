import re
import math
from typing import Callable, Dict, List, Tuple

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from google.genai import types


# ---------------------------------------------------------------------------
# Gemini function-calling tool declarations
# ---------------------------------------------------------------------------

_DECLS: List[types.FunctionDeclaration] = [
    types.FunctionDeclaration(
        name="web_search",
        description="Search the web for current information, news, or research papers.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(type=types.Type.STRING, description="The search query"),
                "max_results": types.Schema(type=types.Type.INTEGER, description="Number of results (default 5)"),
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="fetch_url",
        description="Fetch and read the full text of a web page or article URL.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"url": types.Schema(type=types.Type.STRING, description="The URL to fetch")},
            required=["url"],
        ),
    ),
    types.FunctionDeclaration(
        name="read_file",
        description="Read the contents of a local file.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"filepath": types.Schema(type=types.Type.STRING, description="Path to the file")},
            required=["filepath"],
        ),
    ),
    types.FunctionDeclaration(
        name="add_to_knowledge_base",
        description="Save important text into the RAG knowledge base for future retrieval.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "content": types.Schema(type=types.Type.STRING, description="Text content to store"),
                "source": types.Schema(type=types.Type.STRING, description="Source label or URL"),
            },
            required=["content", "source"],
        ),
    ),
    types.FunctionDeclaration(
        name="calculate",
        description="Evaluate a math expression using Python syntax and the math module.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={"expression": types.Schema(type=types.Type.STRING, description="Math expression")},
            required=["expression"],
        ),
    ),
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _ddg_search(query: str, max_results: int) -> List[Dict]:
    """Core DuckDuckGo search — returns list of {title, url, snippet}. Retries on failure."""
    import time
    for attempt in range(3):
        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", ""),
                    })
            if results:
                return results
        except Exception:
            pass
        if attempt < 2:
            time.sleep(2 ** attempt)   # 1s, 2s backoff
    return results


def web_search(query: str, max_results: int = 5) -> str:
    """Text-only web search (used by MCPToolRegistry)."""
    text, _ = web_search_structured(query, max_results)
    return text


def web_search_structured(query: str, max_results: int = 5) -> Tuple[str, List[Dict]]:
    """
    Returns (formatted_text, results_list) where each result has {title, url, snippet}.
    Used by the streaming agent for richer source display.
    """
    max_results = min(max_results, 10)
    results = _ddg_search(query, max_results)

    if not results:
        return "No results found for this query.", []

    lines = [f"Web search results for: '{query}'\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}\n")

    return "\n".join(lines).strip(), results


def fetch_url(url: str) -> str:
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (ResearchBot/1.0)"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else url
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        text = re.sub(r"\n{3,}", "\n\n", text)
        if len(text) > 8000:
            text = text[:8000] + "\n\n[Content truncated]"
        return f"Title: {title}\nURL: {url}\n\n{text}"
    except Exception as exc:
        return f"Error fetching URL: {exc}"


def read_file(filepath: str) -> str:
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        if len(content) > 8000:
            content = content[:8000] + "\n\n[File truncated]"
        return f"File: {filepath}\n\n{content}"
    except Exception as exc:
        return f"Error reading file: {exc}"


def calculate(expression: str) -> str:
    safe = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
    safe.update({"abs": abs, "round": round, "min": min, "max": max})
    try:
        result = eval(expression, {"__builtins__": {}}, safe)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as exc:
        return f"Calculation error: {exc}"


# ---------------------------------------------------------------------------
# Registry (used by CLI main.py and future tool-call loops)
# ---------------------------------------------------------------------------

class MCPToolRegistry:
    def __init__(self, rag_pipeline=None):
        self._rag = rag_pipeline
        self._handlers: Dict[str, Callable] = {
            "web_search": lambda a: web_search(a["query"], a.get("max_results", 5)),
            "fetch_url": lambda a: fetch_url(a["url"]),
            "read_file": lambda a: read_file(a["filepath"]),
            "calculate": lambda a: calculate(a["expression"]),
        }
        if rag_pipeline is not None:
            self._handlers["add_to_knowledge_base"] = self._add_to_kb

    def _add_to_kb(self, args: Dict) -> str:
        count = self._rag.add_text(args["content"], source=args["source"])
        return f"Stored {count} chunk(s) (source: '{args['source']}')."

    def get_gemini_tool(self) -> types.Tool:
        decls = _DECLS if self._rag else [d for d in _DECLS if d.name != "add_to_knowledge_base"]
        return types.Tool(function_declarations=decls)

    def execute(self, name: str, args: Dict) -> str:
        if name not in self._handlers:
            return f"Unknown tool: '{name}'"
        try:
            return self._handlers[name](args)
        except Exception as exc:
            return f"Tool '{name}' error: {exc}"
