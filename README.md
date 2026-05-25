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
.venv/bin/python ingest.py

# 단일 질의
.venv/bin/python query.py "FACADE 이론이란?"

# 대화형 CLI
.venv/bin/python chat.py

# 에이전트 모드 (멀티스텝 검색 + 웹검색)
.venv/bin/python chat.py --agent
```

---

## 구조

```
인덱싱:
  PDF → MinerU (텍스트 / 이미지 / 표 / 수식)
       → LightRAG KG extraction   gemini-3.5-flash
       → Vision description       gemini-3.1-flash-lite
       → Embedding                gemini-embedding-2 (3072d)

쿼리 (query.py / chat.py):
  질문 → router.py → LightRAG hybrid retrieval → rerank → answer stream
                                                           gpt-5.5 | gemini-3.1-pro-preview

에이전트 쿼리 (chat.py --agent):
  질문 → router.py → tool-calling loop (search_knowledge + web_search_preview)
                   → streaming final answer (동일 대화 이어서)
```

모든 LLM/Embedding 호출은 `httpx` 직접 HTTP — SDK 의존성 없음.

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| 멀티모달 인덱싱 | 텍스트·이미지·표·수식 통합. 인제스트 시 figure 픽셀 기반 캡션 생성 |
| 쿼리 시점 이미지 재주입 | 관련 figure만 관련성 게이트(flash-lite)로 선별해 픽셀 주입 |
| 라우터 pre-filter | flash-lite 1콜로 scope / effort / clarification 판정. fail-open |
| HITL 명료화 게이트 | 모호한 질문은 검색 전 되묻기. 1라운드 상한 |
| LLM reranker | gemini-3.1-flash-lite list-wise 재정렬. `--rerank none|oneshot|batched` |
| 에이전트 루프 | `--agent`: KB tool + web_search_preview tool-calling. `effort=high`시 plan-execute |
| 대화 메모리 | append-only 인용 요약으로 멀티턴 맥락 유지. provider prompt caching 친화적 |
| 세션 영속화 | 매 턴 자동 저장. `--session NAME` / `--resume` |
| Reasoning 스트리밍 | OpenAI `reasoning.summary` delta + Gemini thought parts 동일 인터페이스로 정규화 |
| Arize 트레이싱 | `TRACING=1`으로 활성화. manual OpenInference span (CHAIN/RETRIEVER/LLM/RERANKER) |

---

## CLI 옵션

```bash
# query.py
.venv/bin/python query.py [--provider openai|gemini] [--rerank none|oneshot|batched] "질문"

# chat.py
.venv/bin/python chat.py [--provider openai|gemini] [--rerank none|oneshot|batched]
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
| `config.yaml` | 모델·경로·파서 설정의 단일 출처 |
| `ingest.py` | 문서 인덱싱. WAL 멱등성, `--force` cascade purge |
| `query.py` | 단일 질의 |
| `chat.py` | 멀티턴 CLI. `--agent`로 에이전트 루프 전환 |
| `agent.py` | Responses API tool-calling 루프 + 통합 스트리밍 |
| `kb_tool.py` | LightRAG KB를 OpenAI function tool로 노출 |
| `router.py` | 질의 전 분류기(flash-lite). scope / effort / clarification |
| `image_gate.py` | 쿼리 시점 이미지 관련성 게이트(flash-lite). fail-closed |
| `models.py` | LightRAG 콜백 + provider 라우팅 + style prepend |
| `llm.py` | Gemini/OpenAI HTTP 클라이언트. 비스트리밍 + 스트리밍, reasoning 캡처 |
| `embedding.py` | Gemini batchEmbedContents. 동시성 throttle + 429 백오프 |
| `rerank.py` | LLM 기반 reranker. one-shot / batched |
| `cite.py` | `[src: 문서 \| §섹션 \| p.페이지]` 출처 마커 주입 |
| `tracing.py` | Arize AX OpenInference manual span |
| `prompts/answer_system.md` | 응답 톤·언어·포맷 가이드 (교체 가능) |

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

agent:
  max_tool_rounds: 4              # tool-calling 루프 상한
  web_search: true                # OpenAI 내장 web_search_preview

router:
  model: gemini-3.1-flash-lite
  thinking_budget: {low: 512, medium: 4096, high: -1}
```

---

## 상세 문서

- [설계 결정 노트](docs/design_notes.md) — 요구 → 구현 매핑, 패턴별 근거, 한계
- [에이전트 설계](docs/agentic_design.md) — KB tool 인터페이스, 가치 축 3개, 검증 계획
