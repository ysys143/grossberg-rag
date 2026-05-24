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

쿼리:
  질문
   └─ keyword extraction          gemini-3.5-flash (thinking off)
   └─ KG retrieval                LightRAG hybrid (entities + relations + chunks)
   └─ rerank (선택)               gemini-3.1-flash-lite, list-wise LLM rerank
   └─ prompt assembly             style guide + LightRAG rag_response 템플릿
   └─ answer streaming            openai gpt-5.5 (Responses API + reasoning)
                                  또는 gemini-3.1-pro-preview (thought parts)
```

모든 LLM/Embedding 호출은 **순수 HTTP**로 처리한다. SDK 의존성 없음(`httpx`만).

---

## 2. 모듈 구조와 책임

| 파일 | 역할 |
|------|------|
| `config.yaml` | 모델·경로·파서 설정의 단일 출처 |
| `llm.py` | Gemini/OpenAI HTTP 클라이언트. 비스트리밍 + 스트리밍 변형, 추론 토큰 캡처, `cachedContents`, 사용량 로깅 |
| `embedding.py` | Gemini `batchEmbedContents` 단일 함수 |
| `models.py` | LightRAG가 기대하는 콜백 시그니처에 맞춰 LLM/Vision/Embedding/Answer 함수 조립. provider 라우팅과 style prompt prepending도 여기서 처리 |
| `rerank.py` | LLM 기반 reranker. `rerank()`(one-shot) / `rerank_batched()`(two-stage) |
| `prompts/answer_system.md` | 응답 톤·언어·포맷 가이드(교체 가능) |
| `ingest.py` | 문서 인덱싱 CLI. WAL로 멱등성/충돌 복구 처리 |
| `query.py` | 인터랙티브 + 단일 질의 모드. `only_need_prompt=True`로 retrieval/answer 분리 |
| `ab_test.py` | 동일 질문을 `none`/`oneshot`/`batched` 세 모드로 돌리고 비교 |

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

# 9. 인터랙티브
.venv/bin/python query.py
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
  answer:
    provider: openai              # openai | gemini (CLI --provider로 override)
    openai: gpt-5.5               # Responses API + reasoning stream
    gemini: gemini-3.1-pro-preview
    system_prompt: prompts/answer_system.md

parser:
  engine: mineru
  method: auto
  enable_image: true
  enable_table: true
  enable_equation: true

query:
  default_mode: hybrid            # LightRAG: local | global | hybrid | naive | mix
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
- **멀티턴 대화 미지원**: 인터랙티브 모드는 각 질의가 독립. OpenAI `previous_response_id`/`conversation_id` 도입 시 follow-up 효율적.
- **Reranker accuracy**: 더 다양한 어려운 질의 집합으로 통계적 검증 필요.
- **README가 첫 사용자 친화적이지 않음**: 학습 노트 성격이라 신규 사용자는 별도 quickstart 필요.

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
