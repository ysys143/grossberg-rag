# grossberg-rag

Grossberg *Conscious Mind, Resonant Brain* 4장(62p)의 멀티모달 RAG. 단일 PDF에서 BCS·FCS·FACADE·LAMINART 같은 시각 지각 신경회로 이론을 추출·질의하기 위한 학습용 파이프라인.

본 README는 빌드 결정의 *근거*까지 함께 기록한다 — 어떤 사용자 요구가 어떤 코드 구조로 이어졌는지를 명시한다.

---

## 1. 아키텍처 한눈에

```
인덱싱:
  PDF
   └─ MinerU                      (텍스트 / 이미지 / 표 / 수식 분리)
   └─ LightRAG
        ├─ KG extraction          gemini-3.5-flash (thinking_budget=0,
        │                          + cachedContents 75% 할인)
        ├─ Vision description     gemini-3.1-flash-lite
        └─ Embedding              gemini-embedding-2 (3072d)

쿼리 (query.py / chat.py):
  질문
   └─ router.py                   gemini-3.1-flash-lite (scope / effort / clarification)
   └─ keyword extraction          gemini-3.5-flash (thinking off)
   └─ KG retrieval                LightRAG hybrid (entities + relations + chunks)
   └─ rerank (선택)               gemini-3.1-flash-lite, list-wise LLM rerank
   └─ prompt assembly             style guide + LightRAG rag_response 템플릿
   └─ answer streaming            openai gpt-5.5 (Responses API + reasoning)
                                  또는 gemini-3.1-pro-preview (thought parts)

에이전트 쿼리 (chat.py --agent):
  질문
   └─ router.py                   pre-filter (동일)
   └─ tool-calling loop           OpenAI Responses API, non-streaming (최대 4회)
        ├─ search_knowledge        KBTool → LightRAG (concepts/entities 직접 주입)
        └─ web_search_preview      OpenAI 내장 웹검색
   └─ streaming final answer      동일 대화 컨텍스트 이어서 스트리밍 (별도 호출 없음)
```

모든 LLM/Embedding 호출은 **순수 HTTP**로 처리한다. SDK 의존성 없음(`httpx`만).

---

## 2. 모듈 구조와 책임

| 파일 | 역할 |
|------|------|
| `config.yaml` | 모델·경로·파서 설정의 단일 출처 |
| `llm.py` | Gemini/OpenAI HTTP 클라이언트. 비스트리밍 + 스트리밍 변형, 추론 토큰 캡처, `cachedContents`, 사용량 로깅 |
| `embedding.py` | Gemini `batchEmbedContents`. 동시성 throttle(세마포어) + 429 백오프 + 에러 키 마스킹 |
| `models.py` | LightRAG 콜백(LLM/Vision/Embedding/Answer) 조립. provider 라우팅·style prepend·effort 실현. `vision_model_func`는 RAGAnything의 `image_data`를 forward(§4.20 버그 픽스) |
| `rerank.py` | LLM 기반 reranker. `rerank()`(one-shot) / `rerank_batched()`(two-stage) |
| `prompts/answer_system.md` | 응답 톤·언어·포맷 가이드(교체 가능) |
| `ingest.py` | 문서 인덱싱 CLI. WAL 멱등성/복구, 단계 타이밍 로깅, `--force`는 `adelete_by_doc_id`로 cascade purge(중복 방지). parse→enrich→insert |
| `query.py` | 단일 질의 모드. `only_need_prompt=True`로 retrieval/answer 분리 |
| `chat.py` | 대화형(멀티턴) 검색 CLI. `conversation_history`로 follow-up 맥락 유지 + 라우터/명료화 게이트/이미지 주입 + 런타임 `/provider`·`/rerank`·`/sources` 명령. `--agent`로 에이전트 루프 전환 |
| `kb_tool.py` | LightRAG KB를 OpenAI function tool로 노출. `concepts`→`hl_keywords`, `entities`→`ll_keywords` 직접 주입(내부 키워드 추출 LLM 콜 생략). `only_need_context=True` 고정으로 컨텍스트만 반환. `KBTool` 클래스는 LightRAG 인스턴스를 세션 단위로 재사용해 `initialize_storages()` 1회만 실행 |
| `agent.py` | Responses API tool-calling 루프 + 통합 스트리밍. Stage 1: non-streaming tool-calling(search_knowledge + web_search_preview). Stage 2: 누적된 `input_messages`로 바로 스트리밍(별도 새 호출 없음). `effort=high`면 plan-execute 지시어 주입 |
| `router.py` | 질의 전(前) 분류기(flash-lite). 범위(in_scope)·검색 필요 여부·추론 강도(effort)·명료화 필요 여부를 1콜로 판정. fail-open |
| `image_gate.py` | 검색 후(後) 이미지 관련성 게이트(flash-lite). retrieve된 figure 중 직접 관련된 것만 골라 픽셀 주입 대상으로 선택. fail-closed |
| `ab_test.py` | 동일 질문을 `none`/`oneshot`/`batched` 세 모드로 돌리고 비교 |
| `cite.py` | content_list에 `[src: 문서 \| §섹션 \| p.페이지]` 출처 마커 주입 (page_idx + text_level 활용) |
| `tracing.py` | Arize AX OpenInference 트레이싱. SDK 없는 HTTP 호출용 manual span (LLM/EMBEDDING/RETRIEVER/RERANKER/CHAIN) + 멀티모달 이미지 블록 |
| `measure_caption_divergence.py` | read-only 검증 도구. 저장된 캡션 vs 원본 이미지를 강한 vision 모델로 판정해 발산 점수화(인제스션 로직 아님) |

---

## 3. 사용자 요구가 만든 설계 결정

본 프로젝트는 *요구 → 구현* 매핑을 명시적으로 유지한다.

| # | 사용자 요구 | 적용 결과 |
|---|------------|----------|
| 1 | "RAG-Anything 참고해서 grossberg ch4 처리" | LightRAG + MinerU 기반 파이프라인 시작 |
| 2 | "새로운 repo로 진행" | `/grossberg-rag` 독립 디렉터리, 자체 venv |
| 3 | "apikey.env로부터 키 추가" | `~/.oh-my-zsh/custom/apikey.env` + 프로젝트 `.env` 이중 로드 (override 금지) |
| 4 | "독립적인 의존성 관리 파일" | `pyproject.toml` + `uv` (mineru/transformers 버전 핀 포함) |
| 5 | "임베딩은 gemini embedding 2" → "모델 ID 정확히 확인" | `models/gemini-embedding-2` (3,072d), 위키 확인 후 픽스 |
| 6 | "고추론 분야는 gpt-5.5" | 최종 답변 합성에 OpenAI Responses API |
| 7 | "빠르고 대용량 처리는 gemini-3.1-flash-lite" | Vision description + rerank |
| 8 | "KG 추출에 gpt-5.5는 과해" → "gemini-3.5-flash" | LightRAG 내부 LLM은 flash로 |
| 9 | "config.yaml에 설정 고정" | 모든 가변 값 단일 yaml로 외부화 |
| 10 | "API 호출 SDK 없이 HTTP, llm.py/embedding.py로 추상화" | `httpx` 만 사용, 두 파일이 어댑터 |
| 11 | "멱등성, 중복처리 방지" | 파일 해시 manifest → 이후 WAL로 발전 |
| 12 | "파일 1장으로 스모크 테스트 먼저" | `--pdf` 인자 + 별도 storage 디렉터리 |
| 13 | "Write Ahead Log 같은걸 도입" | `.wal.json`에 `in_progress`/`completed` 상태 머신 + atomic rename |
| 14 | "스토리지 초기화를 왜해?" | 전체 wipe 대신 LightRAG `doc_status`의 해당 doc-id만 surgical 제거 |
| 15 | "로그에 진행시간도 찍히나?" | `time.monotonic()` + WAL `started_at`/`completed_at` 사후 측정 |
| 16 | "llm.py에 middleware로 응답 로깅" | `@with_logging` 데코레이터 → 후에 inline `_log_call`로 진화 |
| 17 | "JSON 블록으로 들어가는지 확인" | LightRAG `kg_query_context` 템플릿 분석, NDJSON 구조 문서화 |
| 18 | "프롬프트 캐싱 정확히 적용" | 단계별 캐싱 가능 영역 식별, usage 통계 캡처 추가 |
| 19 | "conversation_id 필요?" | OpenAI 자동 prefix caching은 별도 — `conversation_id`는 캐싱 아님임을 명시 |
| 20 | "OpenAI도 쓰잖아" (지적) | 누락된 OpenAI 경로 복원 — 최종 답변용 |
| 21 | "최종응답 gpt-5.5로 하라고 했는데" (정정) | `only_need_prompt=True`로 분리 후 별도 호출 |
| 22 | "reasoning step 스트리밍 가능?" | Gemini `streamGenerateContent`(thought parts), OpenAI Responses API(`reasoning.summary` delta) |
| 23 | "모델 선택 가능해야" | `--provider {openai\|gemini}` + `config.yaml: answer.provider` |
| 24 | "Gemini 선택 시 gemini-3.1-pro 사용" | `gemini-3.1-pro-preview`로 ID 보정 |
| 25 | "동일 톤 응답 시스템 프롬프트, markdown 추상화" | `prompts/answer_system.md` + `models.py`가 LightRAG sys_prompt에 prepend |
| 26 | "thinking 토큰 비용 증대" 우려 | `thinking_budget=0`이 가능한 곳에 적용 (keyword/KG) |
| 27 | "Gemini explicit caching 도입" | `cachedContents` API + lock-protected 레지스트리. 인덱싱 시 75% 할인 검증 |
| 28 | "reranker = gemini-3.1-flash-lite LLM rerank" | `rerank.py`. list-wise scoring, JSON parse fallback |
| 29 | "oneshot 정확도 떨어지지 않나?" | `rerank_batched` (two-stage) + A/B 테스트 인프라 |
| 30 | "어려운 질의로 한 번 더 검증" | 대조형 베이지안 비판 질의로 mode별 커버리지 측정 |
| 31 | "commit and push" | Git init, public repo push |
| 32 | "관련/검색 필요 판단 라우터를 앞단에. 답변 모델·effort도" | `router.py` flash-lite 분류기(in_scope/needs_retrieval/effort), provider별 effort 실현은 `models.py` (4.17) |
| 33 | "flash-lite 디폴트로 하되 모델은 config.yaml에서" | `router.model` 설정화, 기본 flash-lite |
| 34 | "모호한 질문이면 멈추고 되묻는 HITL 게이트" | 라우터에 `needs_clarification` 통합(추가 콜 0) + `pending_question` 1라운드 상한 (4.18) |
| 35 | "검색 관련 이미지를 쿼리 시점 컨텍스트로 주입, 불필요하면 가차없이 제외, 인용 명시" | `image_gate.py` 관련성 게이트 + 멀티모달 픽셀 주입 + References `(Figure)` 공개 (4.19) |
| 36 | "쿼리 시점 먼저 구현·검증 후 재인제스트, config 제어, text-only 모델이면 폴백+경고" | `query.inject_images` 토글 + `vision_models` 안전 가드 (4.19) |
| 37 | "처리단계 로깅 로직을 수정해라 / 추측 그만" | `ingest.py` `_stage()` flush 단계 타이밍 → 추측 대신 PARSE/INSERT 직접 계측 (4.20) |
| 38 | "프롬프트나 effort도 의심 / mineru 안 느려" → 측정 결과 진짜 원인 발견 | `vision_model_func`가 `image_data`를 drop하던 배선 버그 픽스 — 캡션이 이미지 기반으로 정확해짐 (4.20) |
| 39 | "--force가 옛 청크를 안 지웠다니" | `adelete_by_doc_id` cascade purge로 교체 → 중복 0 (4.20) |
| 40 | "throttle도 추가 / 동시성 상한 너무 빡세다" | 임베딩 세마포어(기본 8) + 429 백오프 + 키 마스킹 (4.20) |
| 41 | "특정 SDK 의존성 최소화 / 툴은 context만 반환" | `kb_tool.py` httpx-only Responses API. `only_need_context=True` 고정. LangChain 미사용 (4.21) |
| 42 | "LightRAG KB를 tool로 등록, 웹검색도 포함" | `agent.py` tool-calling 루프. `search_knowledge` + `web_search_preview`. `KBTool` lazy init (4.21) |
| 43 | "--agent로 통합" | Stage 2를 별도 새 호출 대신 누적 `input_messages` 이어서 스트리밍. `ANSWER_STYLE_PROMPT` 시스템 메시지에 미리 주입 (4.22) |

---

## 4. 적용된 핵심 패턴

### 4.1 **WAL (Write-Ahead Log) — 멱등성과 충돌 복구**
- `rag_storage/.wal.json` 단일 파일이 파일 SHA-256 해시별 상태(`in_progress` | `completed`) 추적.
- 재실행 시 `in_progress`를 만나면 LightRAG `doc_status`의 stale entry만 surgical 삭제 후 재처리.
- atomic write: tmp 파일 → `os.replace`로 corruption 방지.
- **배운 점**: manifest는 "성공만 기록" 패턴(commit-only)이 비정상 종료에 약함 → WAL 통합으로 *진행 중* 상태도 1급 시민으로.

### 4.2 **HTTP-only Provider Abstraction**
- SDK 의존성 0. `httpx.AsyncClient`만으로 Gemini REST와 OpenAI Chat/Responses API 모두 처리.
- 비-스트리밍/스트리밍 변형 분리. 스트리밍은 `AsyncIterator[{"type": "reasoning"|"answer", "delta": str}]` 정규화 — provider별 SSE 형식 차이를 호출 측에 숨김.
- **배운 점**: 어댑터 패턴이 모델 교체 비용을 만든다 (예: gpt-5.5 → gemini-3.1-pro-preview 토글이 config 1줄).

### 4.3 **Two-Stage Query: LightRAG retrieval + 외부 answer**
- `only_need_prompt=True`로 retrieval + assembled prompt만 받고, 답변 생성은 외부 모델로.
- **이유**: LightRAG의 `llm_model_func`은 KG 추출·키워드 추출에 사용되어 가벼운 모델이 적합하지만, 최종 답변엔 고추론 모델이 필요. 둘을 같은 모델로 묶을 이유가 없음.

### 4.4 **Style Prompt Prepending (Markdown 외부화)**
- 응답 톤/언어/포맷 가이드를 `prompts/answer_system.md`로 외부화. config에서 경로 지정.
- LightRAG가 제공하는 `rag_response` 시스템 프롬프트는 *컨텍스트 사용 규칙*을 담고 있어 교체하면 안 됨 → **prepend** 방식.
- **배운 점**: 프롬프트는 데이터, 코드와 분리. 비개발자 편집·A/B 테스트 용이.

### 4.4b **Consolidation 결정 — NDJSON 그대로 주입 (LLM 요약 단계 없음)**
- "그래프 쿼리 결과를 LLM context에 주입 전 consolidation 거치면 성능이 높아질까?"라는 질문에 대한 답:
  **light consolidation은 유익, heavy consolidation(별도 LLM 요약 단계)은 손해.**
- 검증: LightRAG는 이미 light consolidation (entity/relation dedup, 토큰 truncation, NDJSON 정렬)을 수행.
- LightRAG가 LLM에 주입하는 컨텍스트 형식이 `description` 필드 자연어 + 인용 가능한 `reference_id`를 포함한 NDJSON 블록임을 코드로 확인. **인덱싱 시점에 이미 narrative consolidation이 완료된 상태**.
- 쿼리 시점에 또 LLM 요약을 얹으면 (a) 정보 손실 (b) lost-in-the-middle 역효과 (c) provenance 흐려짐 (d) latency 2배. 그래서 도입하지 않음.

### 4.5 **Provider-Side Prompt Caching**
- **OpenAI**: 자동 prefix caching. `conversation_id` 불필요. 검증 결과 동일 prompt 재사용 시 99% cache hit.
- **Gemini explicit `cachedContents`**: 인덱싱 entity_extraction prompt(5,221 chars)를 캐시 객체로 생성, 200+회 호출에서 재사용. 동시성 처리 위해 hash 키별 `asyncio.Lock`.
- **배운 점**: provider별 캐싱 철학 차이 — OpenAI는 자동·불투명, Google은 명시적·통제 가능. 둘 다 비용 효과 큼.

### 4.6 **Thinking Token 통제**
- `gemini-3.5-flash`는 기본적으로 dynamic thinking 사용. keyword extraction 같은 deterministic 작업엔 thinking이 손해(1.5× prompt 토큰 분량).
- `thinking_budget=0`으로 끔 → 응답 시간 5.5s → 1.5s, 호출당 ~1,000 토큰 절감.
- **배운 점**: reasoning 모드는 항상 좋은 게 아니다. 구조화된 출력엔 over-engineering.

### 4.7 **Reasoning Streaming**
- Gemini: `streamGenerateContent` + `thinkingConfig.includeThoughts=true` → 각 part에 `thought:true` 플래그.
- OpenAI Responses API: `response.reasoning_summary_text.delta` 이벤트로 추론 요약 스트리밍.
- 두 다른 SSE 형식을 `{"type": "reasoning"|"answer", "delta": str}` 단일 인터페이스로 정규화.

### 4.8 **LLM Reranker + A/B Verification**
- list-wise scoring (1회 호출) vs two-stage batched (≤6회). 정확도 vs 비용 trade-off.
- 검증 결과(어려운 베이지안 비판 질의 기준):
  - rerank ON(oneshot)이 분산된 비판 논거(Helmholtz, Kanizsa, shunting network) 회수에 기여
  - batched는 stage-1 cutoff로 V2/V4 청크 false negative 발생, 비용만 증가
- **결론**: oneshot이 기본값. batched는 옵션으로만 유지.

### 4.9 **Observability (사용량·시간 추적)**
- 모든 LLM 호출이 `logs/llm_calls.jsonl`에 append-only로 기록.
- 항목: `fn`, `model`, `prompt`/`response` preview, `usage`(prompt/cached/thoughts/output tokens), `elapsed_s`, `status`.
- JSONL 선택 이유: 동시 append 안전, `jq`/`grep`로 분석 용이.

### 4.10 **점진적 스모크 → 풀 인덱싱**
- 1페이지 → 3페이지 → 62페이지 순서로 비용 누수 없이 검증.
- `--pdf` 인자로 인덱싱·쿼리 시 별도 storage 디렉터리(`rag_storage_{stem}`) 격리.

### 4.11 **Arize AX 트레이싱 (manual OpenInference span)**
- `TRACING=1`로 켜면 인덱싱·쿼리 전 구간이 Arize에 trace로 기록됨.
- SDK 없이 raw HTTP를 쓰므로 auto-instrumentor가 안 잡힘 → `tracing.py`가 로깅 boundary에서 manual span 방출.
- span tree: `CHAIN(query) → RETRIEVER → {EMBEDDING, RERANKER → LLM} → LLM(answer)`, 인덱싱은 `CHAIN(ingest) → {LLM, EMBEDDING, vision}`.
- RETRIEVER span에 검색된 entities/relations/chunks + 주입 컨텍스트, answer span에 주입 프롬프트 전체, vision span에 멀티모달 이미지를 OpenInference 규약으로 첨부.
- **CLI라 flush-before-exit 필수** (비동기 OTLP export가 종료 전 드롭되지 않도록).

### 4.12 **청크 단위 출처 (document · section · page)**
- MinerU는 `page_idx`(페이지) + `text_level`(헤딩)을 보존하지만 RAGAnything가 청킹 시 버림 → references가 파일명으로 collapse.
- `cite.py`가 `parse_document`와 `insert_content_list` 사이에서 각 블록에 `[src: 문서 | §섹션 | p.페이지]` 마커를 주입 → 청크가 출처를 보존.
- 답변 스타일 가이드가 이 마커를 읽어 인라인 인용 + References를 (문서·섹션·페이지) 단위로 생성. 같은 섹션이 여러 페이지면 범위(p.2–3)로 병합.
- **재인덱싱 필수**: 출처 입도는 청크에 저장된 정보에 종속 — 쿼리 시점에 만들 수 없음.

### 4.13 **대화형 메모리 — append-only 인용 요약 (chat.py)**
- `chat.py`는 멀티턴 대화에서 히스토리를 `[Q1, 요약1, Q2, 요약2, ...]` 구조로 누적.
- 각 턴의 assistant 항목 = **그 턴에서 실제 인용된 청크의 1~2문장 요약** (원문 덤프도, 답변 prose도 아님). `gemini-3.1-flash-lite`(thinking off)로 생성하며 `[src:]` 마커를 보존해 출처 추적이 턴을 넘어 유지됨.
- **인용 청크 식별**: 답변이 인용한 페이지를 추출 → 전체 assembled prompt를 스캔해 `[src: ... p.N]` 마커가 인용 페이지와 맞는 유닛만 누적. hybrid/mix 모드에선 Document Chunks 블록이 비고 마커가 엔티티/관계 description에 있으므로 블록 한정이 아닌 전체 스캔이 필요.
- **append-only가 캐싱을 지킴**: 과거 턴 요약은 생성 후 불변 → 프롬프트 prefix 고정 → provider prompt caching 유지. "과거 압축·현재 상세"라는 rolling 요약의 의도는, 현재 턴의 전체 근거를 항상 fresh retrieval이 제공하므로 손실 없이 달성. (rolling은 과거를 재작성해 prefix를 깨므로 채택하지 않음.)
- `conversation_history`는 LightRAG `QueryParam(conversation_history, history_turns)`로 전달돼 retrieval·프롬프트 조립이 맥락을 인지 → follow-up의 지시어("그게")가 이전 턴 주제로 해소됨.

### 4.14 **세션 영속화 (chat.py)**
- 대화 히스토리(append-only 요약)를 **매 턴 자동 저장** → `sessions/<name>.json` (atomic tmp→rename, crash-safe). 종료 시점만이 아니라 턴마다 저장하므로 비정상 종료에도 직전 턴까지 보존 (인덱싱 WAL과 동일 철학).
- `--session NAME`: 명명 세션 재개/생성 · `--resume` (`--continue`/`-c`): 단독이면 가장 최근, `--resume <id>`면 특정 세션 이어가기 · 옵션 없으면 타임스탬프 세션 신규.
- `/sessions`로 저장된 세션 목록(현재 `*`), `/clear`는 초기화 후 저장 반영.
- 히스토리가 요약본이라 세션 파일이 가볍고, 재개해도 캐시 친화적 prefix가 유지됨.

### 4.15 **종료·인터럽트 처리 (chat.py)**
- 종료: `/exit`, plain `exit`/`quit`/`q`, Ctrl-D, 프롬프트에서 Ctrl-C.
- **답변 생성 중 Ctrl-C**: 그 턴만 취소하고 프롬프트로 복귀 (traceback 없음). `__main__`이 stray `KeyboardInterrupt`를 최종 포착해 어느 경로든 clean shutdown.
- 입력 프롬프트의 ANSI 색상은 `\001..\002`(readline 비출력 마커)로 감싸 백스페이스·방향키·히스토리 라인 편집이 정상 동작.

### 4.16 **한글 상태 메시지 (chat.py)**
- LightRAG 영문 INFO 로그를 억제하지 않고 `_KoreanStatusHandler`로 가로채 검색 파이프라인 각 단계를 한글로 재방출: 엔티티/관계 키워드 → 지역/전역 검색 수 → 원시 결과 → 토큰 truncation → 청크 병합 → rerank(M→N) → 최종 컨텍스트. storage-init 노이즈는 패턴 미매칭으로 자동 폐기.

### 4.17 **질의 라우터 (router.py)**
- 검색·답변 같은 비싼 경로 **앞단**에서 flash-lite 1콜로 질문을 분류: `in_scope`(문서 범위/대화 메타 여부) · `needs_retrieval`(검색 필요 여부) · `effort`(low/medium/high) · `needs_clarification`. 라우터는 추상적 `effort`만 내고, provider별 실현(OpenAI `reasoning.effort` vs Gemini `thinking_budget`)은 `models.py`가 담당 — 결정과 실현의 분리.
- 범위 밖이면 검색·답변 모델 호출 없이 즉시 거절(비용 0). 검색 불필요(인사·메타)면 KG 검색을 건너뛰고 대화 히스토리만으로 답변. 답변 모델은 항상 설정된 모델(gpt-5.5), 라우터는 effort만 조절.
- **fail-open**: 라우터가 깨지거나 파싱 실패 시 in_scope + 검색필요 + medium으로 폴백 — 분류기 결함이 비용은 늘려도 정답을 삼키지 않음. 모델은 `config.yaml: router.model`로 교체 가능(기본 flash-lite).

### 4.18 **명료화 HITL 게이트 (chat.py)**
- 라우터가 `needs_clarification`을 함께 판정(추가 콜 0). 질문이 너무 막연하고 대화 히스토리로도 지시어가 안 풀리면 검색하지 않고 한글 되묻기를 출력한 뒤 멈춤. 지시어("그게")가 히스토리로 해소되면 게이트는 작동 안 함(과도한 개입 방지).
- HITL 상태는 `pending_question`으로 유지. 다음 입력을 보충 설명으로 병합해 `skip_clarify=True`로 재실행 — **1라운드 상한**으로 무한 되묻기 방지.

### 4.19 **쿼리 시점 이미지 재주입 (chat.py + image_gate.py)**
- 인제스트 시 figure는 flash-lite가 **텍스트 설명**으로 변환돼 청크가 되고, 그 설명이 검색 키이자 답변 컨텍스트 본문이 됨. (주의: 한때 이 캡션이 색·구조를 틀리는 "캡션 천장" 문제가 있었으나, 그 원인은 vision 모델 한계가 아니라 **§4.20의 배선 버그**였고 수정됨. 지금 캡션은 이미지 기반으로 정확.)
- 그럼에도 깊은 시각 추론(세밀한 공간 배치·화살표 방향 등 캡션이 못 담는 디테일)에는 **원본 픽셀**이 유용. 검색 후 retrieve된 figure 중 **질의와 직접 관련된 것만** `image_gate`(flash-lite)로 선별(≤ `max_injected_images`, 없으면 0개)해 멀티모달 답변 모델에 주입. 관련 없는 figure는 가차없이 제외.
- **fail-closed**: 게이트 오류 시 이미지 0개(텍스트 전용으로 폴백) — 이미지는 보강일 뿐 필수 컨텍스트가 아니므로 라우터와 반대 방향.
- **안전 가드**: `inject_images=true`인데 활성 답변 모델이 `vision_models`에 없으면 주입 취소·텍스트 전용 진행·경고 안내.
- **출처 공개**: 주입된 figure의 `[src:…|image]` 마커를 추적(`last_image_sources`)하고 answer 프롬프트가 References에 `(Figure)`로 명시하도록 지시. `/sources`는 주입된 이미지를 라벨로 표시. 인제스트를 다시 돌리지 않는 query-time 방식(임베딩 불변).

### 4.20 **Vision `image_data` 배선 버그 + 인제스트 견고성**
- **근본 버그(수정됨)**: RAGAnything 모달 프로세서는 figure를 `modal_caption_func(prompt, image_data=<base64>)`로 넘기는데, `models.vision_model_func`는 `images`/`messages`만 처리하고 `image_data`를 `**kwargs`로 **조용히 버렸음.** 그래서 인제스트 내내 모델이 **이미지 없이** 캡션을 생성 → figure 캡션·텍스트 figure 메타가 환각이었음(유명 자극은 캡션 텍스트로 얼추 맞고, 색 부호화 다이어그램은 "흑백"으로 오기). 프레임워크 결함이 아니라 **어댑터 규약 불일치 + `**kwargs` 침묵**이 원인. 수정: `image_data`를 data-URI로 forward → flash-lite가 이미지를 실제로 봄(검증: FIGURE 4.55 색 정확).
- **단계 타이밍 로깅(ingest.py)**: `_stage()`가 PARSE/ENRICH/INSERT 전환을 flush해 찍음. stdout 블록 버퍼링 + 단계 로깅 부재로 "어느 단계가 느린지" 추측하던 문제 해소(예: PARSE=MinerU 235s vs INSERT=vision+KG 811s를 직접 분리).
- **`--force` cascade purge(ingest.py)**: 기존 `_purge`는 `doc_status`만, 그것도 틀린 doc-id로 건드려 **아무것도 안 지웠음** → 재처리 시 옛 청크 위에 새것이 덧쌓여 **중복**(이미지 청크 116=58옛+58새). 수정: parse 후 `lightrag.adelete_by_doc_id(doc_id, delete_llm_cache=True)`로 청크·벡터·엔티티·관계를 연쇄 삭제 후 재삽입 → 재처리가 깔끔히 교체(검증: 재인제스트 결과 58청크, 중복 0).
- **임베딩 throttle(embedding.py)**: 벌크 재인제스트가 수백 임베딩을 동시 호출해 429 다발 → 세마포어 상한(`models.embedding_concurrency`, 기본 8) + 지수 백오프. 429 17→4로 감소. 보너스로 에러 메시지의 `?key=` API 키 **마스킹**(로그 유출 차단).

### 4.21 **에이전트 루프 — KB tool + 웹검색 (agent.py + kb_tool.py)**
- **동기**: 고정 2단계 파이프라인(router → LightRAG → answer)은 멀티홉 검색·웹 보강·검색 전략 선택을 모델이 능동적으로 결정할 수 없음. KB를 **function tool**로 노출해 모델이 언제·어떻게 검색할지 스스로 결정하는 루프로 전환(아키텍처 문서: `docs/agentic_design.md`).
- **세 가지 가치 축**: (1) 웹검색 + 요약 미들웨어(코퍼스 밖 grounding), (2) plan-execute(`effort=high`시 서브질문 분해), (3) 느슨한 결합(KB가 독립 툴 계약으로 분리).
- **SDK 없음**: Responses API tool-calling을 `httpx` 직접 호출로 구현. LangChain 미도입 — 기존 `llm.py` 패턴과 일관성.
- **`KBTool`**: `concepts`(→`hl_keywords`) / `entities`(→`ll_keywords`) 직접 주입으로 내부 키워드 추출 LLM 콜 생략. `only_need_context=True` 고정 — tool은 순수 retriever, agent가 합성·인용 처리. LightRAG 인스턴스를 세션 단위로 재사용(`initialize_storages()` 1회만).
- **웹검색**: OpenAI 내장 `web_search_preview` (추가 패키지/API 키 불필요). Phase-2 TODO: Tavily function tool로 교체 시 요약 미들웨어 적용 가능.
- **plan-execute 게이트**: `effort=high` 질문에만 `_PLAN_INSTRUCTION`을 시스템 메시지에 주입해 서브질문 분해 유도. 단순 질문에 과도 적용 방지.
- **검증**: BCS·FCS·LAMINART·Kanizsa·DaVinci 5개 서브질문 복합 질의에서 Round 1에 3회 병렬 tool-call, Round 2에서 종합 → 15개 인용 포함 5섹션 답변 생성 확인.

### 4.22 **Stage 1·2 통합 — 동일 대화 컨텍스트 이어서 스트리밍**
- **기존 구조**: Stage 1(non-streaming tool-calling) → Stage 2(별도 `answer_model_stream()` 새 호출). Stage 2는 `question + combined_context`만 보며 Stage 1의 추론을 볼 수 없었음.
- **통합 후**: tool-calling 루프 종료 후 누적된 `input_messages`(시스템 프롬프트 + 질문 + 모든 tool call + 모든 tool result)를 그대로 `_responses_stream()`에 전달해 스트리밍. 별도 새 API 호출 불필요.
- **스타일 가이드 주입 위치 이동**: `ANSWER_STYLE_PROMPT`를 Stage 2 직전이 아닌 **시스템 메시지 구성 시점**에 주입 → tool-calling 단계부터 스타일이 컨텍스트에 포함됨.
- **`collected_context` 제거**: 더 이상 tool 결과를 별도로 수집해 재조합할 필요 없음. `input_messages`가 이미 모든 컨텍스트를 담고 있음.
- **배운 점**: Responses API의 tool-calling 대화는 function_call_output 항목이 마지막일 때 streaming 호출로 자연스럽게 이어짐 — 별도 재조합 없이 하나의 일관된 대화.

### 4.23 **BM25 하이브리드 엔티티 시딩 (hybrid_seed.py)**
- **문제**: LightRAG의 `hybrid` 모드는 KG를 쓰지만 **시드 선정이 벡터 kNN 단독**이다 (`operate._get_node_data` → `entities_vdb.query`). 임베딩이 낮게 매긴 정확 명칭·약어·Figure번호(`FACADE`, `BCS`, `Figure 4.25`)는 시드에서 누락되고, 그 엔티티의 서브그래프(관계·이웃·청크) 전체가 컨텍스트에서 사라진다.
- **해결**: 시작 시 엔티티 `content`로 자체 Okapi BM25 인덱스(Lucene식 비음수 idf)를 빌드하고, **`entities_vdb.query`를 인스턴스 단위로 래핑**해 벡터 히트 ∪ BM25 히트(entity_name dedup)를 반환. 래핑이므로 이후 그래프 fetch·엣지 확장은 **변경 없는** `_get_node_data`를 그대로 통과 → BM25 시드도 동일한 그래프 처리를 받는다. 모듈 함수 재구현(브리틀) 회피.
- **결정**: 시드-union(점수 융합 아님) — LightRAG가 순서·토큰버짓·dedup을 계속 소유. `bb25`(Bayesian) 등 score-fusion은 융합을 우리가 떠안아야 해서 배제.
- **토크나이저**: 한국어는 mecab-ko 명사, 영문/Figure번호는 regex. 코퍼스 엔티티명이 대부분 영문이라 lexical 이득의 핵심은 영문 약어·고유명사.
- **안전장치**: `query.hybrid_seed` 기본 off + fail-open(빌드 실패 시 벡터-only). `lightrag-hku==1.4.16` 핀 + 가드 테스트(`_get_node_data`가 `entities_vdb.query`로 시드하는지)로 업그레이드 드리프트를 시끄럽게 실패시킴.
- **한계**: lexical은 동일 언어만 — 한국어 질의는 영문 인덱스에 안 붙는다(→ 4.24가 보완).

### 4.24 **코퍼스 언어 키워드 확장 + 특이 jargon glossary (expand.py)**
- **문제**: 인덱스는 문서 언어(여기선 영문)인데, 타 언어 질의(한국어)는 벡터(임베딩 간극)·BM25(동일 언어만) 양쪽에서 실패. 표준 경로(`engine.ask_events`의 `aquery`)는 `hl/ll_keywords`를 안 넘겨 LightRAG **내부 추출기가 질의 언어 그대로** 키워드를 뽑는다.
- **해결**: flash-lite 1콜로 질의를 **코퍼스 언어 키워드**(concepts=hl, entities=ll)로 확장해 `QueryParam(hl_keywords, ll_keywords)`로 주입 — `kb_tool`이 쓰던 명시 키워드 경로 재사용. `get_keywords_from_query`가 사전 설정 키워드가 있으면 내부 LLM 추출을 스킵함을 확인.
- **병렬화**: router와 `asyncio.gather`로 동시 실행 → 둘 다 cheap flash-lite라 추가 wall-clock ≈ 0. router가 no-retrieval/out-of-scope면 확장 결과는 버림.
- **코퍼스 언어 자동 감지**: `vdb_entities.json` content의 지배 Unicode 스크립트로 판정(ko/en/ja/zh), config 오버라이드. "약어·고유명사 verbatim 보존" 규칙으로 혼합 언어 코퍼스도 견딤.
- **glossary (few-shot 앵커)**: 인덱스에서 **허브 용어(청크빈도 상위) + 특이 jargon(엔티티명 토큰 IDF 상위, df 하한·노이즈 필터)** 혼합을 자동 추출해 확장 프롬프트에 주입 → LLM이 인덱스 정규 표기로 스냅. **콜드 스타트 없음**(인덱스에서 자동 파생), 선택적 수기 overlay는 additive.
- **prefix 캐싱**: glossary를 정렬 고정해 system_prompt를 byte-stable prefix로 만들면 기존 `llm._ensure_gemini_cache`(≥4000자에서 발동)가 `cachedContents`를 생성 → 반복 호출 75% 할인. 질의는 user prompt에만 둬 prefix 불변 유지.
- **검증(A/B)**: 한국어 질의에서 off는 `end-stopped cells`처럼 빗나가고 소문자 변형, on은 `End Cut`·`Filling-In`·`Hypercomplex Cells` 등 **인덱스 정규 표기로 스냅**. 다만 LightRAG 내부 추출기가 이미 일부 영어를 뱉어 이득은 "정밀화"에 가까움.
- **안전장치**: `query.expand_keywords` 기본 off + 모든 단계 fail-open(확장 실패→내부 추출, glossary 실패→빈 glossary).

---

## 5. 설치 및 실행

```bash
# 1. 의존성 (uv 필요)
uv sync

# 2. API 키 설정
#    - ~/.oh-my-zsh/custom/apikey.env 에 OPENAI_API_KEY, GOOGLE_API_KEY
#    - 또는 .env 파일 직접 작성

# 3. config.yaml에서 PDF 경로 지정 (기본: grossberg_ch4.pdf)

# 4. 인덱싱 (62페이지 ≈ 30분, 첫 1회만)
.venv/bin/python ingest.py

# 4-1. 스모크 테스트 (작은 PDF로 검증)
.venv/bin/python ingest.py --pdf grossberg_ch4_p1.pdf

# 5. 질의 (단일)
.venv/bin/python query.py "FACADE 이론이란?"

# 6. provider 토글
.venv/bin/python query.py --provider gemini "질문"
.venv/bin/python query.py --provider openai "질문"

# 7. rerank 모드
.venv/bin/python query.py --rerank none "질문"
.venv/bin/python query.py --rerank oneshot "질문"   # 기본
.venv/bin/python query.py --rerank batched "질문"

# 8. A/B 비교
.venv/bin/python ab_test.py "질문"

# 9. 대화형(멀티턴) 검색 — 세션 영속화, 한글 상태, /명령
.venv/bin/python chat.py                       # 새 세션 (타임스탬프 저장)
.venv/bin/python chat.py --session research1   # 명명 세션 재개/생성
.venv/bin/python chat.py --resume              # 가장 최근 세션 이어가기
.venv/bin/python chat.py --provider gemini --rerank batched
#   세션 중 명령: /provider /rerank /sources /history /sessions /clear /exit

# 10. 에이전트 모드 — 멀티스텝 KB 검색 + 웹검색 (§4.21~4.22)
.venv/bin/python chat.py --agent               # 에이전트 루프 (세션 영속화·히스토리 공유)
.venv/bin/python chat.py --agent --provider openai --session agent1
.venv/bin/python agent.py "질문"               # 단독 실행 (검증용)
```

---

## 6. config.yaml 레퍼런스

```yaml
pdf:
  path: /path/to/document.pdf

storage:
  working_dir: ./rag_storage      # KG, vector DB, WAL
  output_dir: ./output            # MinerU 파싱 캐시

models:
  llm: gemini-3.5-flash           # KG + keyword extraction
  vision: gemini-3.1-flash-lite   # 이미지 설명 + rerank
  embedding: gemini-embedding-2   # 3072-dim
  embedding_dim: 3072
  embedding_max_tokens: 8191
  embedding_concurrency: 8        # 동시 임베딩 호출 상한 (429 가드, §4.20)
  answer:
    provider: openai              # openai | gemini (CLI --provider로 override)
    openai: gpt-5.5               # Responses API + reasoning stream
    gemini: gemini-3.1-pro-preview
    system_prompt: prompts/answer_system.md
    vision_models: [gpt-5.5, gemini-3.1-pro-preview]  # 이미지 입력 가능한 답변 모델

parser:
  engine: mineru
  method: auto
  enable_image: true
  enable_table: true
  enable_equation: true

query:
  default_mode: hybrid            # LightRAG: local | global | hybrid | naive | mix
  inject_images: true             # 쿼리 시점 관련 figure 픽셀 재주입 (4.19)
  max_injected_images: 5          # 관련성 게이트가 주입할 이미지 상한 K
  image_gate_model: gemini-3.1-flash-lite   # 관련 이미지 선별용 분류기

agent:                            # 에이전트 루프 (4.21~4.22, chat.py --agent)
  max_tool_rounds: 4              # tool-calling 반복 안전 상한
  web_search: true                # OpenAI 내장 web_search_preview 포함
                                  # Phase-2 TODO: Tavily function tool로 교체 → 요약 미들웨어 적용 가능

router:                           # 질의 전 분류기 (4.17~4.18)
  enabled: true
  model: gemini-3.1-flash-lite    # 범위·검색필요·effort·명료화 판정
  thinking_budget:                # effort → Gemini thinking_budget (OpenAI은 문자열 직접 사용)
    low: 512
    medium: 4096
    high: -1                      # -1 = dynamic
```

---

## 7. 로그 분석 예시

```bash
# 누적 LLM 호출 통계
jq -s 'group_by(.fn) | map({fn: .[0].fn, count: length, avg_elapsed: (map(.elapsed_s) | add/length)})' logs/llm_calls.jsonl

# 가장 느린 호출 top 5
jq -s 'sort_by(.elapsed_s) | reverse | .[:5]' logs/llm_calls.jsonl

# 캐시 hit ratio (Gemini)
jq -s '[.[] | select(.usage.promptTokenCount)] | {total_prompt: (map(.usage.promptTokenCount) | add), total_cached: (map(.usage.cachedContentTokenCount // 0) | add)}' logs/llm_calls.jsonl
```

---

## 8. 한계와 향후 작업

- **단일 문서 corpus**: 다중 문서 추가 시 reranker의 가치가 크게 올라갈 것으로 예상.
- **Reranker accuracy**: 더 다양한 어려운 질의 집합으로 통계적 검증 필요.
- **웹검색 요약 미들웨어 (Phase-2)**: 현재 `web_search_preview`는 OpenAI 내장으로 결과를 그대로 agent context에 주입. 토큰 폭증·노이즈·출처 혼동 방지를 위해 Tavily function tool + cheap 모델(flash-lite) 요약 미들웨어로 교체 예정. 불변식: URL/출처는 요약 후에도 보존해야 인용 namespace가 유지됨.
- **에이전트 히스토리 품질**: `--agent` 모드의 turn 요약이 `answer[:300]` 단순 절삭. 일반 모드의 `_summarize_cited()`처럼 인용 청크 요약으로 개선하면 멀티턴 맥락 품질이 높아질 것.
- **A/B: 에이전트 vs 고정 파이프라인**: 동일 질의셋을 두 구성으로 돌려 품질·인용 정확도·지연·비용 비교 (`ab_test.py` 확장 또는 별도 스크립트).
- **~~선별 재인제스트(pro 승급)~~ — 불필요로 판명**: 한때 "캡션 천장"을 vision 모델 한계로 보고 복잡 figure만 pro로 재인제스트하려 했으나, 측정 결과 진짜 원인은 §4.20의 `image_data` 배선 버그였음. flash-lite는 **이미지만 제대로 받으면 충분** → pro 승급·복잡도 triage·색 라우터 전부 불필요. (`measure_caption_divergence.py`는 그 회귀 검증용 read-only 도구로 남김 — 단 58장 동시 호출 시 타임아웃이 있어 throttle 보강 필요.)

---

## 9. 사용자 입력 원문 (참고)

이 프로젝트의 모든 설계 결정은 다음 사용자 지시에서 파생되었다(주요 요청만 요약, 단순 ACK·디버그 메시지 제외). 각 항목과 구현의 1:1 매핑은 §3 표 참조:

1. RAG-Anything 참고해서 grossberg ch4 처리
2. LLM/embedding 결정 묻기
3. 새로운 repo로 진행
4. .env에 apikey.env로부터 추가
5. 독립적인 의존성 관리 파일 작성
6. 임베딩 = gemini-embedding-2 (모델 ID 정확히 확인)
7. 고추론 = gpt-5.5
8. 빠르고 대용량 = gemini-3.1-flash-lite
9. KG 추출에 gpt-5.5는 과해 → gemini-3.5-flash
10. config.yaml에 설정 고정
11. SDK 없이 HTTP, llm.py/embedding.py로 추상화
12. 모델 ID에 "2"가 빠짐을 지적
13. 멱등성/중복처리 방지
14. 파일 1장으로 스모크 테스트
15. Write-Ahead Log 도입
16. 스토리지 전체 초기화를 왜? (surgical cleanup 요구)
17. 로그에 진행시간 기록
18. llm.py에 middleware로 응답 로깅
19. JSON 블록 주입 확인
20. 단계별 프롬프트 캐싱 + conversation_id 질문
21. OpenAI도 사용해야 (최종 답변)
22. reasoning step 스트리밍
23. Gemini도 / 모델 선택 가능 / Gemini 시 gemini-3.1-pro
24. 동일 톤 응답 + markdown 추상화
25. thinking 토큰 통제
26. Gemini explicit caching
27. LLM reranker (gemini-3.1-flash-lite)
28. oneshot 정확도 의심
29. 어려운 질의 검증
30. 결론까지
31. commit and push
