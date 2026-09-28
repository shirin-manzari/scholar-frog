"""Small local web interface for the Scholar Frog research pipeline."""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from src.citations import GenerationStatus
from src.generate import generate_answer
from src.index_config import IndexCompatibilityError
from src.retrieve import RetrievalConfig, retrieve
from src.sync import SyncError, sync_library


ROOT = Path(__file__).resolve().parent
PAPERS = ROOT / "papers"
WEB = ROOT / "web"
MAX_PDF_BYTES = 25 * 1024 * 1024


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


def ask_question(question):
    chunks = retrieve(question, top_k=RetrievalConfig.from_env().final_results)
    result = generate_answer(question, chunks)
    cited = result.validation.valid_evidence if result.status is GenerationStatus.ANSWERED else []
    return {
        "status": result.status.value,
        "answer": result.answer,
        "references": [
            {"id": item.evidence_id, "reference": item.reference, "source": item.source,
             "page": item.page, "text": item.text}
            for item in cited
        ],
        "warnings": result.validation.coverage_warnings if cited else [],
    }


class Handler(BaseHTTPRequestHandler):
    def allowed_request(self):
        if self.headers.get("Host") not in {"127.0.0.1:8765", "localhost:8765"}:
            self.send_json({"error": "Use the local Scholar Frog address."}, 403)
            return False
        if self.command == "POST" and self.headers.get("Origin") not in {
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
            path = (PAPERS / relative).resolve()
            if not relative or not path.is_relative_to(PAPERS.resolve()) or path.suffix.lower() != ".pdf" or not path.is_file():
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
                plan = sync_library(str(PAPERS))
                return self.send_json({
                    "added": plan.added, "modified": plan.modified, "deleted": plan.deleted,
                    "unchanged": plan.unchanged, "failures": plan.failures,
                })
            question = payload.get("question", "")
            if not isinstance(question, str) or not question.strip() or len(question) > 4000:
                return self.send_json({"error": "Enter a question of up to 4,000 characters."}, 400)
            return self.send_json(ask_question(question.strip()))
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
