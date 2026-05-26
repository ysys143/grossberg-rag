---
name: grossberg-ask
description: >
  Lets Claude autonomously search the Grossberg Chapter 4 LightRAG knowledge base
  and answer questions about Grossberg neural models — BCS, FCS, FACADE, LAMINART,
  end-stopped cells, bipole cells, illusory contours, neon color spreading, visual
  cortex circuits, and related computational neuroscience topics.

  Use when the user asks anything about: Grossberg Ch4, BCS (Boundary Contour System),
  FCS (Feature Contour System), FACADE, LAMINART, subjective contours, surface
  filling-in, depth perception, visual grouping, neural circuit models, or wants to
  search/query the Grossberg knowledge base.

  Claude runs query_kb.py to retrieve `[src:]`-marked context chunks from LightRAG,
  then synthesizes the answer itself — no external LLM generation step.
---

# Grossberg Ask

## Overview

Retrieve context from the Grossberg Ch4 LightRAG index and answer questions
autonomously. Claude acts as the reasoning agent: search the KB, read the retrieved
chunks (which carry `[src: MARKER]` citation markers), then compose a cited answer.

**Project root:** `~/Documents/GitHub/grossberg-rag`
**Run with:** `uv run python` (from project root, or set `GRAG_DIR`)

---

## Workflow

### 1. Understand the question

Identify:
- Key **entities** — specific model/cell names (BCS, FACADE, LAMINART, end-cut,
  bipole cell, complex cell, V1/V2, etc.)
- Key **concepts** — thematic areas (boundary completion, surface filling-in, depth
  perception, illusory contours, competitive interaction, etc.)
- **Question type** — determines search strategy:

  | Type | Strategy |
  |------|----------|
  | Simple definition | One search, `--mode local` |
  | Mechanism detail | One search, `--mode local` or `hybrid` |
  | Cross-model comparison | Two passes: `hybrid` + `global` |
  | **Enumeration/overview** (e.g., "list all models", "what's in this chapter") | Three passes: (1) name extraction, (2) **model lineage** (`--mode local`), (3) figure list |
  | Multi-hop synthesis | 2-3 searches, `--mode mix` |

> **Enumeration trap**: When asked to list models or summarize a chapter, it is
> tempting to treat name extraction as sufficient. It is not. Model *names* and
> model *relationships* (which extends which, which presupposes which) require
> separate searches. Always run a dedicated lineage search (see example below)
> alongside the name-extraction search.

### 2. Search the KB

Run `query_kb.py` with appropriate flags. Add `--with-images` when the question
involves visual circuits, diagrams, or figures (BCS circuit, FACADE layers, etc.).
Adjust `--mode` based on question type:

| Mode | Best for |
|------|----------|
| `hybrid` (default) | Most questions — balanced local+global |
| `local` | Specific mechanism details, cell-level circuit descriptions |
| `global` | Cross-model comparisons, high-level relationships |
| `mix` | Multi-hop synthesis questions spanning many concepts |

```bash
cd ~/Documents/GitHub/grossberg-rag

# Single query — KG noise stripped, cap auto-computed from mode+query count
uv run python ~/.claude/skills/grossberg-ask/scripts/query_kb.py \
  --query "BCS boundary completion bipole cells" \
  --entities BCS "bipole cell" "end-stopped cell" \
  --concepts "boundary completion" "illusory contour" \
  --mode hybrid

# Multi-query: results merged and deduplicated by [src:] marker
uv run python ~/.claude/skills/grossberg-ask/scripts/query_kb.py \
  --query "BCS end-stopped cells circuit" \
  --query "LAMINART layer 2/3 boundary grouping" \
  --entities BCS LAMINART --mode hybrid

# With figures (circuit diagrams, layer diagrams, etc.)
uv run python ~/.claude/skills/grossberg-ask/scripts/query_kb.py \
  --query "BCS boundary completion bipole cells" \
  --entities BCS "bipole cell" \
  --mode hybrid \
  --with-images

# Override auto cap (use only when the auto value is insufficient)
uv run python ~/.claude/skills/grossberg-ask/scripts/query_kb.py \
  --query "..." --max-chars 30000

# Raw unfiltered output (debugging)
uv run python ~/.claude/skills/grossberg-ask/scripts/query_kb.py \
  --query "..." --full
```

**Output notes:**
- Knowledge Graph entity JSON blobs (no `[src:]`) are stripped — reduces output from
  ~100KB to a fraction, fitting directly in context without file intermediaries
- Output cap is **auto-computed** from `--mode` and query count — no need to set
  `--max-chars` manually:

  | Mode | Base cap | × query scale |
  |------|----------|---------------|
  | `local` | 8,000 | `max(1, n//2+1)` |
  | `hybrid` | 12,000 | same |
  | `global` | 15,000 | same |
  | `mix` | 20,000 | same |

  Use `--max-chars N` only to override when the auto value is insufficient.
- Multi-query results are deduplicated by `[src:]` marker (no repeated chunks)
- `--with-images` appends an `--- IMAGES ---` JSON section; read each file via the
  `Read` tool using the `path` field, then cite using the `marker` field

For complex synthesis questions, run 2-3 searches with varied queries:

```bash
# First pass: local detail
uv run python ~/.claude/skills/grossberg-ask/scripts/query_kb.py \
  --query "LAMINART cortical layer circuit V1 V2" \
  --entities LAMINART V1 V2 "layer 4" "layer 6" \
  --mode local

# Second pass: global relationship
uv run python ~/.claude/skills/grossberg-ask/scripts/query_kb.py \
  --query "BCS FCS FACADE interaction surface perception" \
  --concepts "boundary-surface interface" "filling-in" \
  --mode global
```

### 3. Read the retrieved context

The output contains text chunks with inline citation markers like:
```
... The bipole cell receives excitatory input from cells with the same
orientation ... [src: FIGURE_4_25_p42]
```

Collect all `[src: ...]` markers that directly support your answer.

### 4. Synthesize and cite

Write the answer in Korean (unless the user asked in English). Structure:
- Direct answer first
- Mechanism/circuit details with citations
- Cross-model connections if relevant

Cite every factual claim using the exact marker format: `[src: MARKER]`

---

## Citation Format

Use markers exactly as they appear in the retrieved context:
- Inline: "바이폴 셀은 ... [src: FIGURE_4_25_p42]"
- Sources list at the end if multiple markers used

---

## Environment Setup

The script auto-detects the project root by:
1. `GRAG_DIR` env var (override)
2. Walking up from `cwd` looking for `grag/__init__.py`
3. Fallback: `~/Documents/GitHub/grossberg-rag`

Path resolution uses `grag.paths.DATA_DIR` (single source of truth: `PROJECT_ROOT/data/`).
Figure images live under `data/output*/images/{hash}.*`.

API keys are loaded from `~/.oh-my-zsh/custom/apikey.env` (`OPENAI_API_KEY`, `GOOGLE_API_KEY`).
Never read the key file directly — the project loads it automatically.

---

## Example Search Patterns

**Simple definition:**
```bash
uv run python ~/.claude/skills/grossberg-ask/scripts/query_kb.py \
  --query "What is the FCS (Feature Contour System)?" \
  --entities FCS --mode local
```

**Complex multi-model synthesis:**
```bash
# Run both; combine context before answering
uv run python ~/.claude/skills/grossberg-ask/scripts/query_kb.py \
  --query "BCS FCS FACADE how do they interact" \
  --entities BCS FCS FACADE --concepts "boundary-surface interface" --mode hybrid

uv run python ~/.claude/skills/grossberg-ask/scripts/query_kb.py \
  --query "FACADE invisible boundary perceptual grouping" \
  --concepts "perceptual transparency" "neon spreading" --mode global
```

**Kanizsa / subjective contour:**
```bash
uv run python ~/.claude/skills/grossberg-ask/scripts/query_kb.py \
  --query "Kanizsa triangle subjective contour illusory boundary" \
  --entities "Kanizsa" "bipole cell" "end-cut" \
  --concepts "illusory contour" "boundary completion" --mode mix
```

**Model enumeration / chapter overview (two mandatory passes):**

Pass 1 — name extraction (what models exist):
```bash
uv run python ~/.claude/skills/grossberg-ask/scripts/query_kb.py \
  --query "all models BCS FCS FACADE LAMINART chapter overview" \
  --concepts "model overview" \
  --mode global
```

Pass 2 — lineage / relationships (which extends which):
```bash
# Must use --mode local with development-history keywords.
# global/mix searches are dominated by image captions and miss genealogy prose.
uv run python ~/.claude/skills/grossberg-ask/scripts/query_kb.py \
  --query "model extension development history embedding principle LAMINART 3D FACADE BCS FCS" \
  --entities LAMINART "3D LAMINART" FACADE BCS FCS \
  --concepts "model extension" "embedding principle" "unlumping" \
  --mode local
```

Synthesize both passes before answering. Present models as a hierarchy, not a
flat list. Example structure:
```
BCS/FCS (2D, non-laminar)
  └─ FACADE (3D extension of BCS/FCS, non-laminar)
LAMINART (laminar implementation of BCS/FCS, 2D)
  └─ 3D LAMINART (3D extension of LAMINART = laminar implementation of FACADE)
```
