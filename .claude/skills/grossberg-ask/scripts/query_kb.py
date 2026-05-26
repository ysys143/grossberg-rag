#!/usr/bin/env python3
"""
Query the Grossberg Ch4 LightRAG knowledge base and print retrieved context.

Must be run from the grossberg-rag project root (or set GRAG_DIR env var):
    cd /path/to/grossberg-rag
    uv run python /path/to/query_kb.py --query "BCS boundary completion"

Options:
    --query TEXT        Query (repeatable: --query "A" --query "B")
    --concepts TEXT...  High-level thematic terms (e.g. "boundary completion")
    --entities TEXT...  Specific model names (e.g. BCS FACADE LAMINART)
    --mode MODE         local|global|hybrid|mix  (default: hybrid)
    --working-dir DIR   Override storage dir (default: from config.yaml)
    --with-images       Extract and resolve figure image paths from context
    --max-chars N       Hard cap on total output chars (auto if omitted)
    --full              Disable compaction — emit raw LightRAG output
"""
import argparse
import asyncio
import glob as glob_mod
import json
import os
import re
import sys
from pathlib import Path


def _find_grag_root() -> Path:
    if env := os.environ.get("GRAG_DIR"):
        return Path(env)
    cwd = Path.cwd()
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "grag" / "__init__.py").exists():
            return candidate
    home = Path.home()
    for fallback in [
        home / "Documents/GitHub/grossberg-rag",
        home / "grossberg-rag",
        Path("/workspace/grossberg-rag"),
    ]:
        if (fallback / "grag" / "__init__.py").exists():
            return fallback
    raise FileNotFoundError(
        "Could not find grossberg-rag project. "
        "Run from the project root or set GRAG_DIR=/path/to/grossberg-rag"
    )


def _auto_max_chars(num_queries: int, mode: str) -> int:
    base = {"local": 8_000, "hybrid": 12_000, "global": 15_000, "mix": 20_000}
    return base.get(mode, 12_000) * max(1, num_queries // 2 + 1)


def _compact(context: str, max_chars: int) -> str:
    """Remove Knowledge Graph entity JSON blobs (no [src:] marker) and apply size cap.

    LightRAG prepends a large KG entity section before the actual text chunks.
    Those JSON lines carry no [src:] marker and are not useful for synthesis.
    Keeping only [src:]-bearing chunks typically reduces output from 100KB → 5-15KB.
    """
    kept = []
    for line in context.splitlines():
        s = line.strip()
        # Drop KG entity blobs: JSON objects with no [src:] marker
        if s.startswith("{") and "[src:" not in s:
            continue
        kept.append(line)
    result = "\n".join(kept).strip()
    if len(result) > max_chars:
        result = result[:max_chars] + f"\n\n[...truncated at {max_chars:,} chars]"
    return result


def _merge(results: list[str]) -> str:
    """Combine multiple query results, deduplicating by [src:] marker."""
    seen_markers: set[str] = set()
    merged_lines: list[str] = []
    for result in results:
        current_block: list[str] = []
        block_markers: list[str] = []
        for line in result.splitlines():
            markers = re.findall(r"\[src:[^\[\]]{1,200}\]", line)
            if markers:
                block_markers.extend(markers)
            current_block.append(line)
            # Flush block on blank line
            if not line.strip() and current_block:
                new_markers = [m for m in block_markers if m not in seen_markers]
                if new_markers or not block_markers:
                    merged_lines.extend(current_block)
                    seen_markers.update(block_markers)
                current_block = []
                block_markers = []
        # Flush last block
        if current_block:
            new_markers = [m for m in block_markers if m not in seen_markers]
            if new_markers or not block_markers:
                merged_lines.extend(current_block)
                seen_markers.update(block_markers)
    return "\n".join(merged_lines)


def _resolve_image_path(img_hash: str, content: str, data_dir: Path) -> str | None:
    m = re.search(r"(/\S+?/images/" + re.escape(img_hash) + r"\.\w+)", content)
    if m and Path(m.group(1)).exists():
        return m.group(1)
    hits = glob_mod.glob(
        str(data_dir / "output*" / "**" / "images" / f"{img_hash}.*"),
        recursive=True,
    )
    return hits[0] if hits else None


def _extract_images(context: str, data_dir: Path) -> list[dict]:
    out, seen = [], set()
    for line in context.splitlines():
        s = line.strip()
        if not s.startswith("{"):
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            continue
        content = obj.get("content") or obj.get("description") or ""
        if "Image Path" not in content:
            continue
        hm = re.search(r"images/([a-f0-9]+)", content)
        if not hm or hm.group(1) in seen:
            continue
        img_hash = hm.group(1)
        path = _resolve_image_path(img_hash, content, data_dir)
        if not path:
            continue
        seen.add(img_hash)
        marker_m = re.search(r"\[src:[^\]]*image\]", content)
        marker = marker_m.group(0) if marker_m else ""
        sec_m = re.search(r"§(.+?)\s*\|\s*p\.(\d+)", marker or content)
        fig_m = re.search(r"FIGURE\s+[\d.]+[^\n]*", content)
        out.append({
            "hash": img_hash,
            "path": path,
            "marker": marker,
            "section": sec_m.group(1).strip() if sec_m else "?",
            "page": int(sec_m.group(2)) if sec_m else 0,
            "figure": fig_m.group(0).strip() if fig_m else "",
        })
    return out


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve context from the Grossberg Ch4 KB"
    )
    parser.add_argument("--query", action="append", required=True,
                        metavar="QUERY",
                        help="Search query (repeatable for multi-query)")
    parser.add_argument("--concepts", nargs="*", default=None)
    parser.add_argument("--entities", nargs="*", default=None)
    parser.add_argument("--mode", default="hybrid",
                        choices=["local", "global", "hybrid", "mix"])
    parser.add_argument("--working-dir", default=None)
    parser.add_argument("--with-images", action="store_true")
    parser.add_argument("--max-chars", type=int, default=None,
                        help="Hard cap on output chars (auto-computed if omitted)")
    parser.add_argument("--full", action="store_true",
                        help="Disable compaction; emit raw LightRAG output")
    args = parser.parse_args()

    root = _find_grag_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    os.chdir(root)

    from grag.kb_tool import KBTool  # type: ignore[import]  # noqa: PLC0415
    from grag.paths import DATA_DIR  # type: ignore[import]  # noqa: PLC0415

    kb = KBTool(working_dir=args.working_dir)
    try:
        raw_results = []
        for q in args.query:
            r = await kb.search(
                query=q,
                concepts=args.concepts,
                entities=args.entities,
                mode=args.mode,
                return_answer=False,
            )
            if r:
                raw_results.append(r)

        if not raw_results:
            print("[no results retrieved]")
            return

        # Merge multiple query results (dedup by [src:] marker)
        combined = _merge(raw_results) if len(raw_results) > 1 else raw_results[0]

        # Image extraction runs on raw (pre-compact) context so KG JSON is still present
        images = _extract_images(combined, DATA_DIR) if args.with_images else []

        # Compact unless --full
        max_chars = args.max_chars or _auto_max_chars(len(raw_results), args.mode)
        output = combined if args.full else _compact(combined, max_chars)
        print(output)

        if args.with_images:
            print("\n--- IMAGES ---")
            if images:
                print(json.dumps(images, ensure_ascii=False, indent=2))
            else:
                print("[no figure images found in context]")

    except Exception as exc:  # noqa: BLE001
        print(f"[KB error: {exc}]", file=sys.stderr)
        sys.exit(1)
    finally:
        await kb.close()


if __name__ == "__main__":
    asyncio.run(main())
