"""
Post-hoc citation patch for image chunks — no re-indexing.

Image chunks store "Image Content Analysis: Image Path: .../images/<hash>.jpg".
The MinerU content_list has the same <hash> as img_path plus page_idx, and we can
derive the section from the preceding heading. We map hash -> (section, page) and
prepend a [src: ...] marker to each image chunk's content in BOTH stores
(kv_store_text_chunks.json + vdb_chunks.json). Embeddings are left untouched — the
marker is metadata, not semantic content, so retrieval behavior is unchanged.

Usage: python patch_image_citations.py <working_dir> <content_list.json> <doc_name>
"""
import json
import os
import re
import sys
from pathlib import Path


def build_hash_map(content_list: list[dict]) -> dict[str, tuple[str, int]]:
    """hash -> (section, page) by walking blocks in document order."""
    section = "(front matter)"
    out: dict[str, tuple[str, int]] = {}
    for item in content_list:
        if item.get("type") == "text":
            lvl = item.get("text_level")
            if isinstance(lvl, int) and lvl >= 1 and (item.get("text") or "").strip():
                section = item["text"].strip()
        elif item.get("type") == "image":
            img_path = item.get("img_path", "")
            m = re.search(r"images/([a-f0-9]+)", img_path)
            if m:
                out[m.group(1)] = (section, item.get("page_idx", 0) + 1)
    return out


def _atomic_write(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    os.replace(tmp, path)


def patch_store(path: Path, hmap: dict, doc_name: str, iter_entries) -> int:
    if not path.exists():
        return 0
    data = json.loads(path.read_text())
    patched = 0
    for content_holder in iter_entries(data):
        content = content_holder.get("content", "")
        if "Image Path" not in content or "[src:" in content:
            continue
        m = re.search(r"images/([a-f0-9]+)", content)
        if not m or m.group(1) not in hmap:
            continue
        section, page = hmap[m.group(1)]
        marker = f"[src: {doc_name} | §{section} | p.{page} | image]"
        content_holder["content"] = f"{marker}\n{content}"
        patched += 1
    if patched:
        _atomic_write(path, data)
    return patched


def main():
    if len(sys.argv) < 4:
        print("Usage: python patch_image_citations.py <working_dir> <content_list.json> <doc_name>")
        sys.exit(1)
    wdir = Path(sys.argv[1])
    content_list = json.loads(Path(sys.argv[2]).read_text())
    doc_name = sys.argv[3]

    hmap = build_hash_map(content_list)
    print(f"image hash -> (section,page) map: {len(hmap)} entries")

    # kv_store_text_chunks.json: {chunk_id: {content, ...}}
    n_kv = patch_store(
        wdir / "kv_store_text_chunks.json", hmap, doc_name,
        iter_entries=lambda d: d.values(),
    )
    # vdb_chunks.json: {"data": [{content, vector, ...}], ...}
    n_vdb = patch_store(
        wdir / "vdb_chunks.json", hmap, doc_name,
        iter_entries=lambda d: d.get("data", []),
    )
    print(f"patched: kv_store_text_chunks={n_kv}, vdb_chunks={n_vdb}")


if __name__ == "__main__":
    main()
