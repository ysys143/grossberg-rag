"""
Caption-ceiling measurement (read-only; does NOT modify the index).

For a sample of figures, compare the STORED flash-lite caption against the ACTUAL
original image using a strong vision judge that treats the image as ground truth.
Groups results by figure_type so we can see which figure classes actually diverge
and set the pro-escalation rule from data (instead of blanket re-ingesting all).

Usage: python measure_caption_divergence.py [sample_size]   # default 12
"""
import asyncio
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root -> import grag

from dotenv import load_dotenv

from grag.paths import ENV_PATH, APIKEY_ENV, DATA_DIR

load_dotenv(ENV_PATH)
load_dotenv(APIKEY_ENV, override=False)

from grag import llm

JUDGE = "gemini-3.1-pro-preview"

_PROMPT = """You are auditing an AUTO-GENERATED caption against the ACTUAL figure
image, which is the ground truth.

Auto-generated caption:
\"\"\"{cap}\"\"\"

Compare the caption to the image and report how badly the caption misrepresents
what is actually in the image. Pay special attention to COLOR (color-coded vs
black-and-white), spatial STRUCTURE/layout, arrow/connection directions, and the
NUMBER of panels/elements. Output ONLY JSON, no prose:
{{"figure_type": "diagram|schematic|multipanel|plot|photo|other",
  "divergence": <0-100 int>,
  "has_color_error": <bool>,
  "has_structure_error": <bool>,
  "conflicts": ["<short factual conflict>", ...]}}"""


def _resolve(img_hash: str, content: str) -> str | None:
    m = re.search(r"(/\S+?/images/" + re.escape(img_hash) + r"\.\w+)", content)
    if m and Path(m.group(1)).exists():
        return m.group(1)
    hits = glob.glob(str(DATA_DIR / "output*" / "**" / "images" / f"{img_hash}.*"), recursive=True)
    return hits[0] if hits else None


def _figures() -> list[dict]:
    d = json.loads(Path("rag_storage/kv_store_text_chunks.json").read_text())
    out, seen = [], set()
    for v in d.values():
        c = v.get("content", "")
        if "Image Path" not in c:
            continue
        hm = re.search(r"images/([a-f0-9]+)", c)
        if not hm or hm.group(1) in seen:
            continue
        path = _resolve(hm.group(1), c)
        if not path:
            continue
        seen.add(hm.group(1))
        va = c.find("Visual Analysis")
        cap = c[va:va + 1500] if va >= 0 else c[:1500]
        out.append({"hash": hm.group(1), "cap": cap, "path": path, "chars": len(c)})
    return out


def _sample(figs: list[dict], n: int) -> list[dict]:
    """Spread across caption-length tiers (proxy for complexity)."""
    figs = sorted(figs, key=lambda f: f["chars"], reverse=True)
    if len(figs) <= n:
        return figs
    step = len(figs) / n
    return [figs[int(i * step)] for i in range(n)]


async def _judge(fig: dict) -> dict:
    raw = await llm.generate_with_vision(
        model=JUDGE, prompt=_PROMPT.format(cap=fig["cap"]), images=[fig["path"]]
    )
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    obj = json.loads(m.group(0)) if m else {}
    obj["hash"] = fig["hash"]
    return obj


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    figs = _sample(_figures(), n)
    print(f"judging {len(figs)} figures with {JUDGE}...\n")
    results = await asyncio.gather(*(_judge(f) for f in figs))

    by_type: dict[str, list[int]] = {}
    for r in results:
        t = r.get("figure_type", "other")
        by_type.setdefault(t, []).append(int(r.get("divergence", 0)))
        flags = []
        if r.get("has_color_error"):
            flags.append("COLOR")
        if r.get("has_structure_error"):
            flags.append("STRUCT")
        print(f"  [{r.get('divergence', '?'):>3}] {t:11s} {r['hash'][:10]} "
              f"{','.join(flags) or '-':14s} {'; '.join(r.get('conflicts', [])[:2])[:90]}")

    print("\n=== divergence by figure_type (mean / n) ===")
    for t, ds in sorted(by_type.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {t:11s} mean={sum(ds) / len(ds):5.1f}  n={len(ds)}  {ds}")


if __name__ == "__main__":
    asyncio.run(main())
