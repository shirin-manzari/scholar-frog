import argparse
import sys
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.generate import generate_answer
from src.ingest import ingest_folder, reingest_folder
from src.retrieve import RETRIEVAL_MODES, RetrievalConfig, retrieve
from src.sync import SyncError, sync_library

console = Console()


def sync_main(argv=None):
    parser = argparse.ArgumentParser(description="Synchronize the paper library and search index.")
    parser.add_argument("--papers", default="papers", help="Paper library directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview without changing the index")
    parser.add_argument("--force", action="store_true",
                        help="Allow deletion batches above SYNC_DELETE_THRESHOLD")
    args = parser.parse_args(argv)
    try:
        with console.status("[bold cyan]Scanning paper library...[/bold cyan]"):
            plan = sync_library(args.papers, dry_run=args.dry_run, force=args.force)
    except SyncError as exc:
        console.print(f"[red]Synchronization stopped:[/red] {exc}")
        return 1

    title = "Library synchronization preview" if args.dry_run else "Library synchronization"
    console.print(f"\n[bold]{title}[/bold]")
    for label, count in (
        ("Added", plan.added), ("Modified", plan.modified), ("Renamed", plan.renamed),
        ("Deleted", plan.deleted), ("Unchanged", plan.unchanged),
        ("Duplicate copies", plan.duplicated), ("Failed", len(plan.failures)),
    ):
        console.print(f"{label + ':':<20}{count}")
    if plan.operations:
        console.print("\n[bold]Planned operations:[/bold]" if args.dry_run else "\n[bold]Changes:[/bold]")
        for operation in plan.operations:
            console.print(f"  {operation}")
    if plan.failures:
        console.print("\n[red]Failures:[/red]")
        for failure in plan.failures:
            console.print(f"  • {failure}")
    if args.dry_run:
        console.print("\n[dim]No changes applied.[/dim]")
    return 1 if plan.failures else 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "sync":
        return sync_main(sys.argv[2:])

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
    parser.add_argument("--citation-retries", type=int,
                        default=int(__import__("os").getenv("CITATION_MAX_RETRIES", "1")),
                        help="Maximum citation repair generations (default: 1)")
    parser.add_argument("--no-citation-validation", action="store_true",
                        help="Disable citation reference validation")
    parser.add_argument("--no-citation-coverage", action="store_true",
                        help="Disable conservative uncited sentence warnings")
    parser.add_argument("--debug-citations", action="store_true",
                        help="Show evidence mapping, validation details, and original response")
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")
    if args.citation_retries < 0:
        parser.error("--citation-retries must be non-negative")

    if not args.ingest_only and not args.question:
        parser.error('a question is required unless --ingest-only is set')
    if args.reingest and args.no_ingest:
        parser.error("--reingest cannot be used with --no-ingest")

    if not args.no_ingest or args.ingest_only or args.reingest:
        console.print("[bold]Checking for new papers to ingest...[/bold]")
        try:
            if args.reingest:
                reingest_folder(args.papers)
            else:
                ingest_folder(args.papers)
        except SyncError as exc:
            console.print(f"[red]Synchronization stopped:[/red] {exc}")
            return 1

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
            result = generate_answer(args.question, chunks, max_retries=args.citation_retries,
                                     coverage_enabled=not args.no_citation_coverage,
                                     validation_enabled=not args.no_citation_validation)
    except (RuntimeError, ImportError, ValueError) as exc:
        console.print(f"[red]Generation failed:[/red] {exc}")
        return 1

    console.print(Panel(Markdown(result.answer), title="Answer", border_style="green"))
    if result.validation.references_valid:
        console.print("\n[bold]References:[/bold]")
        for item in result.validation.valid_evidence:
            console.print(f"  [{item.evidence_id}] {item.reference}")
    if result.validation.coverage_warnings:
        console.print("\n[yellow]Possible uncited factual sentences (heuristic):[/yellow]")
        for sentence in result.validation.coverage_warnings:
            console.print(f"  • {sentence}")
    if args.debug_citations:
        console.print("\n[bold]Citation validation:[/bold] " + ("valid references; semantic support not checked" if result.validation.references_valid else "FAILED"))
        console.print(f"Regeneration attempts: {result.regeneration_attempts}")
        for error in result.error_messages:
            console.print(f"  [red]{error}[/red]")
        for item in result.evidence:
            console.print(f"  [{item.evidence_id}] chunk={item.chunk_id} {item.reference}")
        if not result.validation.references_valid:
            console.print("Original generated response:")
            console.print(result.original_answer)


if __name__ == "__main__":
    raise SystemExit(main() or 0)
