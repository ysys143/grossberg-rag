"""
Citation enrichment for content_list.

MinerU preserves page_idx + text_level (heading marker) per block, but RAGAnything
drops them when it joins blocks into chunk text — so references collapse to the bare
filename. We re-inject a structured source marker into each block BEFORE insertion so
every chunk carries (document, section, page), and the answer model can cite them.

Marker format (kept compact + parseable):
    [src: <doc> | §<section> | p.<page>]
"""
from pathlib import Path


def _is_heading(item: dict) -> bool:
    # MinerU sets text_level (>=1) on heading blocks
    lvl = item.get("text_level")
    return isinstance(lvl, int) and lvl >= 1


def enrich_content_list(content_list: list[dict], doc_name: str) -> list[dict]:
    """Return a copy of content_list with a [src: ...] marker prepended to each
    text/multimodal block, tracking the current section from heading blocks."""
    section = "(front matter)"
    out: list[dict] = []

    for item in content_list:
        new = dict(item)
        page = item.get("page_idx", 0) + 1  # page_idx is 0-based
        ctype = item.get("type", "text")

        if ctype == "text":
            text = item.get("text", "") or ""
            if _is_heading(item) and text.strip():
                section = text.strip()
            marker = f"[src: {doc_name} | §{section} | p.{page}]"
            new["text"] = f"{marker}\n{text}" if text else marker
        else:
            # multimodal (image/table/equation): the modal processor builds its own
            # chunk from the vision description and reads image_caption/img_caption
            # (modalprocessors.py:233), ignoring the item's "text". So inject the
            # marker into the caption list it actually surfaces.
            marker = f"[src: {doc_name} | §{section} | p.{page} | {ctype}]"
            injected = False
            for cap_field in ("image_caption", "img_caption", "table_caption"):
                if cap_field in item:
                    caps = item.get(cap_field) or []
                    if isinstance(caps, list):
                        new[cap_field] = [marker, *caps]
                    else:
                        new[cap_field] = [marker, str(caps)]
                    injected = True
                    break
            if not injected:
                new["img_caption"] = [marker]  # image items default to img_caption
            # keep text marker too, for any text-path fallback
            existing = item.get("text", "")
            new["text"] = f"{marker}\n{existing}" if existing else marker

        out.append(new)

    return out


def doc_name_for(pdf_path: Path) -> str:
    return pdf_path.name
