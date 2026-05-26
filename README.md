# grossberg-rag

Grossberg *Conscious Mind, Resonant Brain* 4장(62p) 멀티모달 RAG.  
BCS·FCS·FACADE·LAMINART 시각 지각 신경회로를 LightRAG 지식 그래프로 인덱싱하고, 대화형 CLI와 에이전트 루프로 질의한다.

---

## 빠른 시작

```bash
# 의존성 설치 (uv 필요)
uv sync

# API 키 설정 (~/.oh-my-zsh/custom/apikey.env 또는 .env)
# OPENAI_API_KEY, GOOGLE_API_KEY

# 인덱싱 (62페이지, 첫 1회만 — 약 30분)
.venv/bin/python -m grag.ingest

# 단일 질의
.venv/bin/python -m grag.query "FACADE 이론이란?"

# 대화형 CLI
.venv/bin/python -m grag.cli

# 에이전트 모드 (멀티스텝 검색 + 웹검색)
.venv/bin/python -m grag.cli --agent

# 웹 앱 (CLI와 동일 엔진 + figure 이미지 표시)
.venv/bin/uvicorn grag.server:app   # http://127.0.0.1:8000
```

> 모든 명령은 **저장소 루트에서** 실행한다 (`grag` 패키지가 import되도록).
> 런타임 데이터(인덱스·파싱 출력·세션·로그·PDF)는 전부 `data/` 아래에 있다.

---

## 구조

```
인덱싱:
  PDF → MinerU (텍스트 / 이미지 / 표 / 수식)
       → LightRAG KG extraction   gemini-3.5-flash
       → Vision description       gemini-3.1-flash-lite
       → Embedding                gemini-embedding-2 (3072d)

쿼리 (grag.query / grag.cli / grag.server):
  질문 → router → LightRAG hybrid retrieval → rerank → answer stream
                                                       gpt-5.5 | gemini-3.1-pro-preview

에이전트 쿼리 (grag.cli --agent):
  질문 → router → tool-calling loop (search_knowledge + web_search_preview)
                → streaming final answer (동일 대화 이어서)
```

CLI와 웹 서버는 동일한 이벤트 엔진(`grag/engine.py`)을 소비한다 — 로직은 엔진에,
표현만 소비자(CLI는 stdout, 웹은 SSE+이미지)에 둬서 양쪽이 절대 어긋나지 않는다.
모든 LLM/Embedding 호출은 `httpx` 직접 HTTP — SDK 의존성 없음.

### 디렉토리 레이아웃

```
grag/            패키지 (소스 + prompts/ + static/)
  paths.py       모든 경로의 단일 출처 (config·data·assets)
  engine.py      이벤트 yield 채팅 엔진 (CLI·웹 공용)
  cli.py server.py   진입점 (python -m grag.cli / uvicorn grag.server:app)
scripts/         일회성 분석 스크립트 (ab_test, measure_*, patch_*)
tests/           pytest 스위트
docs/            설계 노트
data/            런타임 데이터 (gitignored): rag_storage/ output/ sessions/ logs/ pdfs/
config.yaml      모델·경로·파서 설정의 단일 출처
```

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| 멀티모달 인덱싱 | 텍스트·이미지·표·수식 통합. 인제스트 시 figure 픽셀 기반 캡션 생성 |
| 쿼리 시점 이미지 재주입 | 관련 figure만 관련성 게이트(flash-lite)로 선별해 픽셀 주입 |
| 라우터 pre-filter | flash-lite 1콜로 scope / effort / clarification 판정. fail-open |
| HITL 명료화 게이트 | 모호한 질문은 검색 전 되묻기. 1라운드 상한 |
| LLM reranker | gemini-3.1-flash-lite list-wise 재정렬. `--rerank none|oneshot|batched` |
| BM25 하이브리드 시딩 | LightRAG의 벡터-only 엔티티 시드에 BM25 lexical 시드를 union(`query.hybrid_seed`). 약어·Figure번호·고유명사 등 임베딩이 놓치는 정확매칭 복원. mecab-ko 토크나이저 |
| 코퍼스 언어 키워드 확장 | 질의를 코퍼스 언어로 확장해 `hl/ll_keywords` 주입(`query.expand_keywords`). router와 병렬 실행, fail-open. 코퍼스 특이 jargon glossary를 prompt에 few-shot 주입(`query.expand_glossary`, Gemini prefix 캐싱) |
| 에이전트 루프 | `--agent`: KB tool + web_search_preview tool-calling. `effort=high`시 plan-execute |
| 대화 메모리 | append-only 인용 요약으로 멀티턴 맥락 유지. provider prompt caching 친화적 |
| 세션 영속화 | 매 턴 자동 저장. `--session NAME` / `--resume` |
| Reasoning 스트리밍 | OpenAI `reasoning.summary` delta + Gemini thought parts 동일 인터페이스로 정규화 |
| Arize 트레이싱 | `TRACING=1`으로 활성화. manual OpenInference span (CHAIN/RETRIEVER/LLM/RERANKER) |

---

## CLI 옵션

```bash
# 단일 질의
.venv/bin/python -m grag.query [--provider openai|gemini] [--rerank none|oneshot|batched] "질문"

# 대화형 CLI
.venv/bin/python -m grag.cli [--provider openai|gemini] [--rerank none|oneshot|batched]
                             [--agent]
                             [--session NAME] [--resume [ID]]

# 세션 중 명령
/provider <name>   답변 모델 전환
/rerank <mode>     rerank 모드 전환
/sources           마지막 답변의 인용 출처
/history           대화 히스토리
/sessions          저장된 세션 목록
/clear             히스토리 초기화
/exit              종료
```

---

## 모듈

| 파일 | 역할 |
|------|------|
| `config.yaml` | 모델·경로·파서 설정의 단일 출처 (저장소 루트) |
| `grag/paths.py` | config·data·assets 경로의 단일 출처 |
| `grag/ingest.py` | 문서 인덱싱. WAL 멱등성, `--force` cascade purge |
| `grag/query.py` | 단일 질의 |
| `grag/cli.py` | 멀티턴 CLI. `--agent`로 에이전트 루프 전환 |
| `grag/server.py` | FastAPI 웹 서버. engine 이벤트를 SSE로 + figure 이미지 |
| `grag/engine.py` | 이벤트 yield 채팅 엔진 (CLI·웹 공용 단일 출처) |
| `grag/agent.py` | Responses API tool-calling 루프 + 통합 스트리밍 |
| `grag/kb_tool.py` | LightRAG KB를 OpenAI function tool로 노출 |
| `grag/router.py` | 질의 전 분류기(flash-lite). scope / effort / clarification |
| `grag/hybrid_seed.py` | BM25 lexical 엔티티 시드를 LightRAG 벡터 시드에 union (mecab-ko + 자체 Okapi BM25) |
| `grag/expand.py` | 코퍼스 언어 키워드 확장 + 특이 jargon glossary 자동 추출 (prefix-cache 친화) |
| `grag/image_gate.py` | 쿼리 시점 이미지 관련성 게이트(flash-lite). fail-closed |
| `grag/models.py` | LightRAG 콜백 + provider 라우팅 + style prepend |
| `grag/llm.py` | Gemini/OpenAI HTTP 클라이언트. 비스트리밍 + 스트리밍, reasoning 캡처 |
| `grag/embedding.py` | Gemini batchEmbedContents. 동시성 throttle + 429 백오프 |
| `grag/rerank.py` | LLM 기반 reranker. one-shot / batched |
| `grag/cite.py` | `[src: 문서 \| §섹션 \| p.페이지]` 출처 마커 주입 |
| `grag/tracing.py` | Arize AX OpenInference manual span |
| `grag/prompts/answer_system.md` | 응답 톤·언어·포맷 가이드 (교체 가능) |
| `grag/static/index.html` | 웹 앱 단일 페이지 (vanilla JS, 빌드 없음) |

---

## config.yaml 핵심 항목

```yaml
models:
  llm: gemini-3.5-flash           # KG + keyword extraction
  vision: gemini-3.1-flash-lite   # figure 캡션 + rerank
  embedding: gemini-embedding-2
  answer:
    provider: openai              # openai | gemini
    openai: gpt-5.5
    gemini: gemini-3.1-pro-preview
    system_prompt: prompts/answer_system.md

query:
  default_mode: hybrid
  inject_images: true             # 쿼리 시점 figure 재주입
  max_injected_images: 5
  hybrid_seed: false              # BM25 lexical 엔티티 시드 union (needs python-mecab-ko)
  hybrid_seed_top_k: 10
  expand_keywords: false          # 질의 -> 코퍼스 언어 hl/ll 키워드 확장 후 주입
  expand_lang: auto               # 인덱스에서 코퍼스 언어 자동 감지 (또는 en|ko|ja|zh 고정)
  expand_glossary: true           # 코퍼스 특이 용어 glossary를 확장 프롬프트에 주입 (prefix-cached)

agent:
  max_tool_rounds: 4              # tool-calling 루프 상한
  web_search: true                # OpenAI 내장 web_search_preview

router:
  model: gemini-3.1-flash-lite
  thinking_budget: {low: 512, medium: 4096, high: -1}
```

---

## Claude Code 스킬 (`grossberg-ask`)

Claude Code에서 `grossberg-ask` 스킬로 KB를 직접 질의할 수 있다.
스킬이 `query_kb.py`를 호출해 `[src:]` 마커가 붙은 컨텍스트 청크를 가져오고,
Claude가 직접 종합·인용 답변을 작성한다 — 별도 LLM 생성 단계 없음.

```bash
# 스킬 설치 경로
~/.claude/skills/grossberg-ask/scripts/query_kb.py

# 단일 질의 (local 모드 → 자동 캡 8,000자)
uv run python ~/.claude/skills/grossberg-ask/scripts/query_kb.py \
  --query "bipole cell boundary completion" \
  --entities "bipole cell" BCS --mode local

# 멀티 질의 + mix 모드 (3쿼리 → 자동 캡 40,000자)
uv run python ~/.claude/skills/grossberg-ask/scripts/query_kb.py \
  --query "BCS FACADE overview" \
  --query "LAMINART cortical layers" \
  --query "figure-ground filling-in" \
  --mode mix

# figure 이미지 포함
uv run python ~/.claude/skills/grossberg-ask/scripts/query_kb.py \
  --query "BCS circuit diagram" --mode hybrid --with-images

# 원시 출력 (디버깅)
uv run python ~/.claude/skills/grossberg-ask/scripts/query_kb.py \
  --query "..." --full
```

**출력 캡 자동 계산** — `--max-chars` 미지정 시 모드와 쿼리 수에서 자동 결정:

| 모드 | 기본 캡 | 스케일 (쿼리 n개) |
|------|---------|-----------------|
| `local` | 8,000 | `× max(1, n//2+1)` |
| `hybrid` | 12,000 | 동일 |
| `global` | 15,000 | 동일 |
| `mix` | 20,000 | 동일 |

`--max-chars N`은 자동값이 부족한 예외 상황에만 사용한다.

---

## 상세 문서

- [설계 결정 노트](docs/design_notes.md) — 요구 → 구현 매핑, 패턴별 근거, 한계
- [에이전트 설계](docs/agentic_design.md) — KB tool 인터페이스, 가치 축 3개, 검증 계획
