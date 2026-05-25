"""
Grossberg Ch4 — document ingestion with Write-Ahead Log (WAL)

Idempotency model:
  rag_storage/.wal.json tracks per-file state by SHA-256 hash.

  Lifecycle:
    (no entry)      → fresh run, write 'in_progress'
    'in_progress'   → previous run was interrupted (kill/crash/battery)
                      → auto-recover by purging LightRAG's stale doc_status
                      → retry the run, keeping WAL in 'in_progress'
    'completed'     → already successfully indexed → SKIP (unless --force)

Atomic writes: WAL is written via tmp-file + rename so a crash mid-write
cannot leave the WAL itself in a corrupt state.

Usage:
  python ingest.py [--force] [--pdf path/to/file.pdf]
"""
import asyncio
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure venv bin is in PATH so subprocess can find `mineru`
_venv_bin = str(Path(sys.executable).parent)
os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path.home() / ".oh-my-zsh/custom/apikey.env", override=False)

import yaml
from raganything import RAGAnything
from raganything.config import RAGAnythingConfig

from models import llm_model_func, vision_model_func, embedding_func
from cite import enrich_content_list, doc_name_for
import tracing

tracing.init_tracing("grossberg-rag")
_tracer = tracing.get_tracer()

_cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())

_WAL = ".wal.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_elapsed(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def _wal_load(working_dir: Path) -> dict:
    p = working_dir / _WAL
    return json.loads(p.read_text()) if p.exists() else {}


def _wal_save(working_dir: Path, wal: dict) -> None:
    """Atomic write: tmp file → rename, so a crash mid-write cannot corrupt WAL."""
    target = working_dir / _WAL
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(wal, indent=2))
    os.replace(tmp, target)


def _wal_begin(working_dir: Path, file_hash: str, pdf_path: Path) -> bool:
    """Record intent to process. Returns True if this is a recovery (resumed run)."""
    wal = _wal_load(working_dir)
    is_recovery = wal.get(file_hash, {}).get("status") == "in_progress"

    wal[file_hash] = {
        "status": "in_progress",
        "file_path": str(pdf_path),
        "name": pdf_path.name,
        "size_bytes": pdf_path.stat().st_size,
        "started_at": _now(),
        "completed_at": None,
    }
    _wal_save(working_dir, wal)
    return is_recovery


def _wal_complete(working_dir: Path, file_hash: str) -> None:
    wal = _wal_load(working_dir)
    if file_hash in wal:
        wal[file_hash]["status"] = "completed"
        wal[file_hash]["completed_at"] = _now()
        _wal_save(working_dir, wal)


def _purge_lightrag_doc_status(working_dir: Path, file_hash: str) -> None:
    """Remove this doc's entry from LightRAG's internal doc_status.

    Targets only the affected doc-id; other documents' indices are preserved.
    """
    doc_status_path = working_dir / "kv_store_doc_status.json"
    if not doc_status_path.exists():
        return

    data = json.loads(doc_status_path.read_text())
    doc_id = f"doc-{file_hash}"
    if doc_id in data:
        del data[doc_id]
        doc_status_path.write_text(json.dumps(data, indent=2))


def _stage(msg: str) -> None:
    """Flushed, timestamped stage marker so progress is visible in real time even
    when stdout is redirected to a file (block-buffered)."""
    print(f"[{datetime.now():%H:%M:%S}] STAGE: {msg}", flush=True)


async def ingest(force: bool = False, pdf_override: Path | None = None):
    pdf_path = pdf_override or Path(_cfg["pdf"]["path"])
    suffix = f"_{pdf_path.stem}" if pdf_override else ""
    working_dir = Path(_cfg["storage"]["working_dir"] + suffix)
    output_dir = Path(_cfg["storage"]["output_dir"] + suffix)
    p = _cfg["parser"]

    if not pdf_path.exists():
        print(f"ERROR: PDF not found — {pdf_path}")
        sys.exit(1)

    working_dir.mkdir(exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_hash = _sha256(pdf_path)
    wal = _wal_load(working_dir)
    entry = wal.get(file_hash, {})
    status = entry.get("status")

    if status == "completed" and not force:
        print(f"SKIP: {pdf_path.name} already indexed at {entry['completed_at']}")
        print("      Use --force to reprocess.")
        return

    is_recovery = _wal_begin(working_dir, file_hash, pdf_path)
    if is_recovery:
        print(f"RECOVERY: previous run was interrupted (WAL status=in_progress)")
        print(f"          purging stale LightRAG doc_status for doc-{file_hash[:12]}...")
        _purge_lightrag_doc_status(working_dir, file_hash)
    elif force and status == "completed":
        # --force on a completed run: clear stale state to ensure clean reprocess
        _purge_lightrag_doc_status(working_dir, file_hash)

    config = RAGAnythingConfig(
        working_dir=str(working_dir),
        parser=p["engine"],
        parse_method=p["method"],
        parser_output_dir=str(output_dir),
        enable_image_processing=p["enable_image"],
        enable_table_processing=p["enable_table"],
        enable_equation_processing=p["enable_equation"],
    )

    rag = RAGAnything(
        config=config,
        llm_model_func=llm_model_func,
        vision_model_func=vision_model_func,
        embedding_func=embedding_func,
    )

    if not rag.check_parser_installation():
        print("ERROR: MinerU not installed. Run: uv pip install 'raganything[all]'")
        sys.exit(1)

    t_wall = datetime.now().strftime("%H:%M:%S")
    print(f"[{t_wall}] Processing: {pdf_path.name}  (sha256: {file_hash[:12]}...)")
    print("Multimodal parsing in progress (images / tables / equations)...")

    t0 = time.monotonic()
    # CHAIN span around indexing so vision (multimodal) LLM spans get a parent
    # and are exported. Each image description call nests under this.
    import contextlib
    ingest_cm = (
        _tracer.start_as_current_span("ingest") if _tracer else contextlib.nullcontext()
    )
    with ingest_cm as ispan:
        if ispan is not None:
            ispan.set_attribute("openinference.span.kind", "CHAIN")
            ispan.set_attribute("input.value", pdf_path.name)
            ispan.set_attribute("metadata.sha256", file_hash)
        # parse → enrich with (document, section, page) source markers → insert.
        # Splitting parse/insert lets us inject citation metadata into each block
        # before LightRAG chunks it (process_document_complete would skip this).
        # Per-stage timing (flushed) so a slow run shows WHICH stage is slow —
        # parse(MinerU) vs insert(vision desc + KG extract + embed) — instead of guessing.
        _stage("PARSE start (MinerU)")
        t = time.monotonic()
        content_list, doc_id = await rag.parse_document(
            file_path=str(pdf_path),
            output_dir=str(output_dir),
            parse_method=p["method"],
        )
        _stage(f"PARSE done in {time.monotonic() - t:.1f}s — {len(content_list)} blocks, "
               f"{sum(1 for b in content_list if b.get('type') == 'image')} images")

        _stage("ENRICH (inject citation markers)")
        content_list = enrich_content_list(content_list, doc_name_for(pdf_path))

        _stage("INSERT start (vision desc + KG extract + embed)")
        t = time.monotonic()
        await rag.insert_content_list(
            content_list,
            file_path=pdf_path.name,
            doc_id=doc_id,
            display_stats=True,
        )
        _stage(f"INSERT done in {time.monotonic() - t:.1f}s")
    elapsed = time.monotonic() - t0

    _wal_complete(working_dir, file_hash)
    print(f"Done. Elapsed: {_fmt_elapsed(elapsed)}  ({elapsed:.1f}s)")
    print(f"      WAL: {working_dir / _WAL}")


if __name__ == "__main__":
    force = "--force" in sys.argv
    pdf_arg = next(
        (sys.argv[i + 1] for i, a in enumerate(sys.argv) if a == "--pdf" and i + 1 < len(sys.argv)),
        None,
    )
    try:
        asyncio.run(ingest(force=force, pdf_override=Path(pdf_arg) if pdf_arg else None))
    finally:
        tracing.shutdown_tracing()  # flush spans before exit (CLI)
