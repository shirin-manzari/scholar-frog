import argparse
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.generate import generate_answer
from src.ingest import ingest_folder, reingest_folder
from src.retrieve import RETRIEVAL_MODES, RetrievalConfig, retrieve

console = Console()


def main():
    parser = argparse.ArgumentParser(description="Ask a question across your PDF papers.")
    try:
        config = RetrievalConfig.from_env()
    except ValueError as exc:
        parser.error(str(exc))

    parser.add_argument("question", nargs="?", help="The question to ask.")
    parser.add_argument("--papers", default="papers", help="Folder of PDFs (default: ./papers)")
    parser.add_argument("--top-k", type=int, default=config.final_results,
                        help="Number of chunks to retrieve")
    parser.add_argument("--retrieval", choices=RETRIEVAL_MODES, default=config.mode,
                        help="Retrieval mode (default: from RETRIEVAL_MODE or hybrid-rerank)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare dense, hybrid, and hybrid-rerank retrieval")
    parser.add_argument("--no-ingest", action="store_true",
                         help="Skip the (fast, idempotent) ingestion check before querying")
    parser.add_argument("--ingest-only", action="store_true",
                         help="Just ingest the papers folder, don't ask anything")
    parser.add_argument("--reingest", action="store_true",
                         help="Reprocess all PDFs, including unchanged files")
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")

    if not args.ingest_only and not args.question:
        parser.error('a question is required unless --ingest-only is set')
    if args.reingest and args.no_ingest:
        parser.error("--reingest cannot be used with --no-ingest")

    if not args.no_ingest or args.ingest_only or args.reingest:
        console.print("[bold]Checking for new papers to ingest...[/bold]")
        if args.reingest:
            reingest_folder(args.papers)
        else:
            ingest_folder(args.papers)

    if args.ingest_only:
        return

    console.print(f"\n[bold]Q:[/bold] {args.question}\n")

    if args.compare:
        for mode in RETRIEVAL_MODES:
            start = time.perf_counter()
            results = retrieve(args.question, top_k=args.top_k, retrieval_mode=mode)
            elapsed = time.perf_counter() - start
            console.print(f"[bold cyan]{mode}[/bold cyan] — {elapsed:.2f}s")
            if not results:
                console.print("  No results.")
            for result in results:
                scores = "  ".join(
                    f"{name}={result[name]:.4f}"
                    for name in (
                        "dense_score", "bm25_score", "rrf_score", "reranker_score"
                    )
                    if result[name] is not None
                )
                console.print(
                    f"  {result['id']} | {result['title']} — {result['section']} "
                    f"— p.{result['page']} ({result['source']}) | {scores}"
                )
                console.print(f"    {result['text'][:400].strip()}\n")
        return

    with console.status("[bold cyan]Retrieving relevant excerpts...[/bold cyan]"):
        chunks = retrieve(
            args.question, top_k=args.top_k, retrieval_mode=args.retrieval
        )

    if not chunks:
        console.print("[yellow]No relevant excerpts found. Have you added PDFs to the papers folder?[/yellow]")
        return

    try:
        with console.status("[bold cyan]Generating grounded answer...[/bold cyan]"):
            answer = generate_answer(args.question, chunks)
    except (RuntimeError, ImportError, ValueError) as exc:
        console.print(f"[red]Generation failed:[/red] {exc}")
        return 1

    console.print(Panel(Markdown(answer), title="Answer", border_style="green"))

    console.print("\n[bold]Retrieved sources:[/bold]")
    seen = set()
    for c in chunks:
        key = (c["source"], c["page"])
        if key not in seen:
            seen.add(key)
            console.print(f"  • {c['title']} — p.{c['page']}  [dim]({c['source']})[/dim]")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
