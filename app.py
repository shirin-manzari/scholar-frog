"""Small local web interface for the Scholar Frog research pipeline."""

import json
import os
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from src.citations import GenerationStatus
from src.generate import generate_answer
from src.index_config import IndexCompatibilityError
from src.retrieve import RetrievalConfig, retrieve
from src.sync import SyncDeletionConfirmationRequired, SyncError, sync_library
from src.text_normalize import evidence_excerpt_for_display


ROOT = Path(__file__).resolve().parent
PAPERS = ROOT / "papers"
WEB = ROOT / "web"
MAX_PDF_BYTES = 25 * 1024 * 1024
MISSING_PAPER_MESSAGES = (
    "Paper moved or renamed. frog searched the whole swamp.",
    "That PDF vanished. i blame the wizards.",
)


def upload_filename(header):
    """Decode a browser-supplied filename without allowing a filesystem path."""
    try:
        name = unquote(header, errors="strict")
    except UnicodeDecodeError:
        return None
    if (not name or len(name.encode("utf-8")) > 200
            or any(char in name for char in "/\\:")
            or any(ord(char) < 32 or ord(char) == 127 for char in name)
            or Path(name).suffix.lower() != ".pdf"
            or not name[:-4].strip(" .")):
        return None
    return name


def library_status():
    papers_root = PAPERS.resolve()
    papers = sorted(
        (path.relative_to(PAPERS).as_posix() for path in PAPERS.rglob("*")
         if path.is_file() and path.suffix.lower() == ".pdf"
         and path.resolve().is_relative_to(papers_root)),
        key=str.casefold,
    )
    return {"papers": papers}


def paper_path(relative):
    """Resolve a library-relative PDF path without allowing traversal."""
    if not isinstance(relative, str) or not relative:
        return None
    path = (PAPERS / relative).resolve()
    if (not path.is_relative_to(PAPERS.resolve()) or path.suffix.lower() != ".pdf"
            or not path.is_file()):
        return None
    return path


def _available_chunks(chunks, paper_paths):
    available = []
    for chunk in chunks:
        metadata = chunk.get("metadata") or {}
        aliases = metadata.get("source_paths", [])
        if isinstance(aliases, str):
            try:
                aliases = json.loads(aliases)
            except json.JSONDecodeError:
                aliases = []
        if not isinstance(aliases, list):
            aliases = []
        source = next((path for path in [chunk.get("source"), *aliases]
                       if isinstance(path, str) and path in paper_paths), None)
        if source is None:
            continue
        if source != chunk.get("source"):
            chunk = {**chunk, "source": source, "metadata": {**metadata, "source": source}}
        available.append(chunk)
    return available


def ask_question(question, paper=None):
    paper_paths = set(library_status()["papers"])
    if not paper_paths:
        return {"status": "no_papers", "answer": "", "references": [], "warnings": []}
    if paper is not None and paper not in paper_paths:
        raise ValueError(random.choice(MISSING_PAPER_MESSAGES))
    config = RetrievalConfig.from_env()
    chunks = _available_chunks(
        retrieve(question, top_k=config.final_results, paper=paper), paper_paths
    )
    result = generate_answer(question, chunks)
    cited = result.validation.valid_evidence if result.status is GenerationStatus.ANSWERED else []
    return {
        "status": result.status.value,
        "answer": result.answer,
        "references": [
            {"id": item.evidence_id, "reference": item.reference, "source": item.source,
             "page": item.page, "text": evidence_excerpt_for_display(item.text),
             "passage": (item.metadata.get("chunk_index", -1) + 1
                         if isinstance(item.metadata.get("chunk_index"), int) else None),
             "character_start": item.metadata.get("character_start"),
             "character_end": item.metadata.get("character_end")}
            for item in cited
        ],
        "warnings": result.validation.coverage_warnings if cited else [],
        "semantic_support": result.validation.semantic_support,
        "semantic_verdicts": getattr(result.validation, "semantic_verdicts", []),
    }


class Handler(BaseHTTPRequestHandler):
    def allowed_request(self):
        if self.headers.get("Host") not in {"127.0.0.1:8765", "localhost:8765"}:
            self.send_json({"error": "Use the local Scholar Frog address."}, 403)
            return False
        if self.command in {"POST", "DELETE"} and self.headers.get("Origin") not in {
            None, "http://127.0.0.1:8765", "http://localhost:8765"
        }:
            self.send_json({"error": "Cross-site requests are not allowed."}, 403)
            return False
        return True

    def send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_file(self, path, content_type):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.allowed_request():
            return
        parsed = urlsplit(self.path)
        if parsed.path == "/api/status":
            return self.send_json(library_status())
        if parsed.path == "/api/paper":
            relative = parse_qs(parsed.query).get("path", [""])[0]
            path = paper_path(relative)
            if path is None:
                return self.send_json({"error": "Paper not found."}, 404)
            return self.send_file(path, "application/pdf")
        static = {
            "/": (WEB / "index.html", "text/html; charset=utf-8"),
            "/css/styles.css": (WEB / "css/styles.css", "text/css; charset=utf-8"),
            "/js/app.js": (WEB / "js/app.js", "application/javascript; charset=utf-8"),
            "/assets/scholar-frog-idle.png": (WEB / "assets/scholar-frog-idle.png", "image/png"),
            "/assets/scholar-frog-talking.png": (WEB / "assets/scholar-frog-talking.png", "image/png"),
            "/assets/scholar-frog-crying.png": (WEB / "assets/scholar-frog-crying.png", "image/png"),
        }
        if parsed.path in static:
            path, content_type = static[parsed.path]
            return self.send_file(path, content_type)
        self.send_json({"error": "Not found."}, 404)

    def do_DELETE(self):
        if not self.allowed_request():
            return
        parsed = urlsplit(self.path)
        if parsed.path != "/api/paper":
            return self.send_json({"error": "Not found."}, 404)
        relative = parse_qs(parsed.query).get("path", [""])[0]
        path = paper_path(relative)
        if path is None:
            return self.send_json({"error": "Paper not found."}, 404)
        try:
            path.unlink()
        except OSError as exc:
            return self.send_json({"error": f"Could not remove the PDF: {exc}"}, 422)
        return self.send_json({
            "deleted": relative,
            "message": "PDF removed. Sync the library to remove it from the search index.",
        })

    def do_POST(self):
        if not self.allowed_request():
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self.send_json({"error": "Invalid request length."}, 400)
        if length <= 0 or length > MAX_PDF_BYTES:
            return self.send_json({"error": "Request must be between 1 byte and 25 MB."}, 413)
        parsed = urlsplit(self.path)
        if parsed.path == "/api/upload":
            name = upload_filename(self.headers.get("X-Filename", ""))
            if name is None:
                return self.send_json({"error": "Choose a PDF filename without path separators or control characters."}, 400)
            data = self.rfile.read(length)
            if not data.startswith(b"%PDF-"):
                return self.send_json({"error": "This file does not appear to be a PDF."}, 400)
            PAPERS.mkdir(exist_ok=True)
            target = PAPERS / name
            stem = target.stem
            number = 2
            while True:
                try:
                    with target.open("xb") as output:
                        output.write(data)
                    break
                except FileExistsError:
                    target = PAPERS / f"{stem} ({number}).pdf"
                    number += 1
            return self.send_json({"filename": target.name}, 201)
        if parsed.path not in {"/api/sync", "/api/ask"}:
            return self.send_json({"error": "Not found."}, 404)
        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object.")
            if parsed.path == "/api/sync":
                force = payload.get("force", False)
                if not isinstance(force, bool):
                    return self.send_json({"error": "force must be a boolean."}, 400)
                plan = sync_library(str(PAPERS), force=force)
                return self.send_json({
                    "added": plan.added, "modified": plan.modified, "deleted": plan.deleted,
                    "unchanged": plan.unchanged, "failures": plan.failures,
                })
            question = payload.get("question", "")
            if not isinstance(question, str) or not question.strip() or len(question) > 4000:
                return self.send_json({"error": "Enter a question of up to 4,000 characters."}, 400)
            paper = payload.get("paper")
            if paper is not None and (not isinstance(paper, str) or not paper):
                return self.send_json({"error": "Select a valid paper."}, 400)
            return self.send_json(ask_question(question.strip(), paper=paper))
        except SyncDeletionConfirmationRequired as exc:
            return self.send_json({
                "error": str(exc), "code": "delete_confirmation_required",
                "deleted": exc.deleted, "indexed": exc.indexed,
            }, 409)
        except (json.JSONDecodeError, ValueError) as exc:
            return self.send_json({"error": str(exc)}, 400)
        except (SyncError, IndexCompatibilityError, RuntimeError, ImportError) as exc:
            return self.send_json({"error": str(exc)}, 422)


def main():
    os.chdir(ROOT)
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("Scholar Frog is running at http://127.0.0.1:8765")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
