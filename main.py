import os
import sys

from google import genai
from google.genai import types
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from langsmith import traceable
from rag import RAGPipeline
from mcp_tools import MCPToolRegistry

load_dotenv()

_api_key = os.getenv("GEMINI_API_KEY")
if not _api_key:
    sys.exit("ERROR: GEMINI_API_KEY not found in .env")

_client = genai.Client(api_key=_api_key)

GENERATION_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_TOOL_ITERATIONS = 10

SYSTEM_PROMPT = """You are an expert AI Research Assistant. Help users research any topic deeply and accurately.

Your capabilities:
1. **RAG Knowledge Base** — Retrieved context from ingested documents is injected into your prompt. Always cite [Source N] when using it.
2. **web_search** — Search the web for current information, papers, or news.
3. **fetch_url** — Read the full text of any web page or article.
4. **read_file** — Read a local file on the user's machine.
5. **add_to_knowledge_base** — Persist important content for future retrieval.
6. **calculate** — Evaluate mathematical expressions.

Guidelines:
- Use tools proactively when the question requires fresh or detailed information.
- Cite RAG sources as [Source N] and web results with their URL.
- Be accurate and structured. Use Markdown headings, bullets, and code blocks when helpful.
- After a web_search, consider fetch_url on the most relevant result for full detail.
- If unsure, say so and suggest how to find out."""


class ResearchAssistant:
    def __init__(self):
        self.console = Console()
        self.rag = RAGPipeline()
        self.tools = MCPToolRegistry(rag_pipeline=self.rag)
        self._model_name = GENERATION_MODEL
        self._chat = self._new_chat()

    def _new_chat(self):
        return _client.chats.create(
            model=self._model_name,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[self.tools.get_gemini_tool()],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )

    # ------------------------------------------------------------------
    # Agentic query loop: RAG context injection + MCP tool execution
    # ------------------------------------------------------------------

    @traceable(run_type="chain", name="CLI Research Query", tags=["cli"])
    def query(self, user_input: str) -> str:
        rag_chunks = self.rag.search(user_input)
        if rag_chunks:
            context = self.rag.build_context(rag_chunks)
            message = (
                f"**User question:** {user_input}\n\n"
                f"**Retrieved context from knowledge base:**\n{context}\n\n"
                "Answer using the context above and any additional tools you need."
            )
        else:
            message = user_input

        response = self._chat.send_message(message)

        for _ in range(MAX_TOOL_ITERATIONS):
            fn_calls = [
                part.function_call
                for candidate in response.candidates
                for part in candidate.content.parts
                if hasattr(part, "function_call")
                and part.function_call
                and part.function_call.name
            ]

            if not fn_calls:
                break

            fn_parts = []
            for fc in fn_calls:
                name = fc.name
                args = dict(fc.args)
                first_val = repr(list(args.values())[0]) if args else ""
                self.console.print(
                    f"  [dim]↳ tool:[/dim] [cyan]{name}[/cyan] [dim]{first_val}[/dim]",
                    markup=True,
                )
                result = self.tools.execute(name, args)
                fn_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=name, response={"result": result}
                        )
                    )
                )

            response = self._chat.send_message(fn_parts)

        # Extract final text
        try:
            return response.text
        except Exception:
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        return part.text
            return "[No text response generated]"

    # ------------------------------------------------------------------
    # Source ingestion
    # ------------------------------------------------------------------

    def ingest_source(self, source: str) -> str:
        try:
            if source.startswith("http://") or source.startswith("https://"):
                count = self.rag.add_url(source)
                return f"Added {count} chunk(s) from URL: {source}"
            elif os.path.exists(source):
                count = self.rag.add_file(source)
                return f"Added {count} chunk(s) from file: {source}"
            else:
                count = self.rag.add_text(source, source="manual_input")
                return f"Added {count} chunk(s) from text input"
        except Exception as exc:
            return f"Error ingesting source: {exc}"

    # ------------------------------------------------------------------
    # Interactive CLI
    # ------------------------------------------------------------------

    def run(self):
        self.console.print(
            Panel.fit(
                "[bold blue]AI Research Assistant[/bold blue]\n"
                f"[dim]Gemini {self._model_name}  ·  RAG (ChromaDB)  ·  MCP Tools[/dim]\n\n"
                "[yellow]Commands[/yellow]\n"
                "  [cyan]/add <url | filepath | text>[/cyan]  — ingest into knowledge base\n"
                "  [cyan]/kb[/cyan]                          — show knowledge base size\n"
                "  [cyan]/clear[/cyan]                       — wipe knowledge base\n"
                "  [cyan]/model <name>[/cyan]                — switch Gemini model\n"
                "  [cyan]/exit[/cyan]                        — quit\n\n"
                "[dim]Any other input is sent as a research query.[/dim]",
                title="Welcome",
                border_style="blue",
            )
        )

        while True:
            try:
                user_input = Prompt.ask("\n[bold green]You[/bold green]").strip()
                if not user_input:
                    continue

                if user_input.lower() in ("/exit", "/quit", "exit", "quit"):
                    self.console.print("[dim]Goodbye![/dim]")
                    break

                elif user_input.startswith("/add "):
                    src = user_input[5:].strip()
                    with self.console.status("Ingesting…", spinner="dots"):
                        msg = self.ingest_source(src)
                    self.console.print(f"[green]{msg}[/green]")

                elif user_input == "/kb":
                    n = self.rag.total_chunks()
                    self.console.print(f"[cyan]Knowledge base: {n} chunk(s)[/cyan]")

                elif user_input == "/clear":
                    self.rag.clear()
                    self.console.print("[yellow]Knowledge base cleared.[/yellow]")

                elif user_input.startswith("/model "):
                    self._model_name = user_input[7:].strip()
                    self._chat = self._new_chat()
                    self.console.print(f"[cyan]Switched to model: {self._model_name}[/cyan]")

                else:
                    self.console.print()
                    with self.console.status("Thinking…", spinner="dots"):
                        answer = self.query(user_input)
                    self.console.print("[bold blue]Assistant[/bold blue]\n")
                    self.console.print(Markdown(answer))

            except KeyboardInterrupt:
                self.console.print("\n[dim]Tip: type /exit to quit[/dim]")
            except Exception as exc:
                self.console.print(f"[red]Error: {exc}[/red]")


def main():
    ResearchAssistant().run()


if __name__ == "__main__":
    main()
