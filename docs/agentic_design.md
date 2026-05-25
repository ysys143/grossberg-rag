# Agentic RAG 구상 — LangChain `create_agent` + LightRAG 검색 툴

> 상태: **구상(draft)**. 구현 전 합의용 문서. 코드 아님.
> 대상: 현재 고정 2단계 파이프라인을 agent 기반으로 전환할지 검토.

## 1. 동기 — 무엇을 바꾸려는가

현재 grossberg-rag는 **고정 2단계 파이프라인**이다:

```
질문 → router.py(분류) → LightRAG 검색+프롬프트 조립(only_need_prompt) → 외부 답변 모델(gpt-5.5) → 인용 정리
```

한계: 검색을 *언제/어떤 전략으로* 할지가 사전에 고정된다. 멀티홉(검색→부족하면 재검색), 코퍼스 밖 질문(웹), 질문별 검색 mode 최적화를 파이프라인이 능동적으로 못 한다.

전환 목표: LightRAG 지식베이스를 **툴**로 노출하고, [LangChain `create_agent`](https://reference.langchain.com/python/langchain/agents/factory/create_agent)의 model(=gpt-5.5)이 **언제·어떻게 검색할지 스스로 결정**하는 agentic loop로 바꾼다.

### 핵심 가치 3축

전환의 가치는 세 개의 직교하는 차원으로 정리된다. (중요: 이 셋은 "mode 라우팅이 항상-hybrid를 이기는가"라는 불확실한 측정에 **의존하지 않는다** — 가치 근거가 구조 자체에 있음. §8 참조.)

| 축 | 차원 | 내용 |
|---|---|---|
| **#1 웹검색 + 요약 미들웨어** | 능력(무엇으로 grounding) | 코퍼스 밖 질문을 웹으로 보강하되, 결과를 **요약 미들웨어**로 압축해 agent 컨텍스트 오염·토큰 폭증을 막는다. 웹(무한·노이즈)을 길들이는 장치 = web_search를 안전하게 추가하는 전제. |
| **#2 plan-execute** | 추론(얼마나 깊이) | 복합 질문을 계획→실행→부족 시 재조사. **조건부**: 모든 질문에 적용하면 과함. `router.py`의 `effort`가 게이트 — `high`만 plan-execute, `low/medium`은 단순 ReAct. |
| **#3 느슨한 결합** | 구조(어떻게) | KB를 깨끗한 툴 계약으로 떼어내 검색+답변 강결합(현 `query.py`/`chat.py`)을 해소. KB가 독립 테스트·교체·재사용(추후 MCP 노출) 가능한 서비스가 됨. |

교차 관심사: **#1의 요약 미들웨어가 출처(URL)를 버리면 §4 인용 분리가 무너진다.** "요약하되 출처 보존"이 단일 실패점.

```
질문 → create_agent(model=gpt-5.5)
         ├─ tool: search_knowledge  (LightRAG 검색기)
         └─ tool: web_search        (코퍼스 밖/최신/검증)
       → 모델이 tool_calls 발행 → 그래프가 실행 → ToolMessage 반환 → 반복 → 최종 답변
```

`create_agent`의 루프: 모델이 `AIMessage.tool_calls`를 내면 그래프가 툴을 실행하고 `ToolMessage`로 결과를 붙여 더 이상 호출이 없을 때까지 반복.

## 2. 툴 인터페이스

### 2.1 `search_knowledge` (LightRAG 검색기)

LightRAG `QueryParam`이 아래를 **네이티브 지원**함을 확인했다(`lightrag/base.py`):
- `hl_keywords: list[str]` / `ll_keywords: list[str]` — 키워드 직접 주입(내부 키워드 추출 LLM 콜 생략)
- `only_need_context: bool` — 답변 생성 없이 검색 컨텍스트만 반환
- `mode: local|global|hybrid|naive|mix|bypass`
- `conversation_history` — 멀티턴 메모리가 툴 경유로도 흐름

```python
@tool
def search_knowledge(
    query: str,                  # 답변/재랭킹용 자연어 질의
    concepts: list[str] = [],    # → hl_keywords: 주제/메커니즘 (global 성격)
    entities: list[str] = [],    # → ll_keywords: 특정 용어/엔티티 (local 성격)
    mode: Literal["local","global","hybrid","mix"] = "hybrid",
    return_answer: bool = False, # False=컨텍스트만, True=LightRAG가 답변까지
) -> str: ...
```

두 가지 호출 모드(arg로 선택):
1. **컨텍스트 조회** (`return_answer=False`, `only_need_context=True`): 인용 마커(`[src: doc|§|p.]`)가 박힌 raw 컨텍스트를 반환. agent(gpt-5.5)가 직접 종합.
2. **답변까지** (`return_answer=True`): LightRAG 내부 LLM(gemini)이 답변 생성. 빠른 사실 조회용.

### 2.2 `web_search`

코퍼스(62p Grossberg ch4) 밖 질문, 최신 논문, 사실 검증용. 책 인용과 **출처 네임스페이스가 다름**(URL vs `[src:]`).

```python
@tool
def web_search(query: str) -> str: ...   # 내부적으로 요약 미들웨어 경유
```

**요약 미들웨어(가치 축 #1)**: 원본 웹 결과는 길고 노이즈가 많아 그대로 agent 컨텍스트에 넣으면 토큰 폭증 + 신호 희석 + 멀티홉 시 누적 오염. cheap 모델(flash-lite)로 질의 관련 추출만 압축. 구현 위치 두 갈래:
- (a) `web_search` 툴 내부에서 압축 후 반환 (단순)
- (b) ToolMessage를 가로채는 `AgentMiddleware` (LightRAG 결과에도 재사용 가능)

**불변식**: 요약은 반드시 **URL/출처를 보존**해야 함 — 버리면 §4 인용 분리가 무너지는 단일 실패점.

## 3. 의도 라우팅 전략

핵심 아이디어: **별도 분류기 없이 툴 인자 구성 자체가 라우팅**이 된다. agent가 질문을 읽고

- 주제/종합 질문 → `concepts` 채움, `mode="global"` (예: "시각 인지의 핵심 메커니즘")
- 특정 용어/정의 → `entities` 채움, `mode="local"` (예: "BCS가 뭐야")
- 복합/다중 시스템 비교 → 둘 다 채움, `mode="hybrid"`
- 코퍼스 밖 → `web_search`
- 인사/메타/이전 답변 재질문 → 툴 미호출(또는 `mode="bypass"`)

### 기존 `router.py`와의 관계

`router.py`는 이미 front-of-pipeline 분류기로 `in_scope` / `needs_retrieval` / `effort` / `needs_clarification` / `clarification`을 산출한다. agentic 전환 시 선택지:

- **(A) 유지 — 프리필터**: router를 agent 앞단 게이트로 남겨, 범위 밖/잡담/모호질문을 *agent 호출 전에* 싸게 거른다(flash-lite). 인스코프 질문만 agent로. → 토큰 절약 + HITL clarification 보존.
- **(B) 흡수 — middleware**: `AgentMiddleware.before_model`로 옮겨 라우팅을 agent 그래프 안에 둔다. 깔끔하지만 LangChain 기계장치 증가, clarification HITL 재설계 필요.

권장: **(A)**. 이미 동작하는 router를 프리필터로 두고, 통과한 질문만 agent가 도구를 들고 처리. 두 레이어 책임이 명확히 분리됨(router=진입 게이트, agent=검색 전략·멀티홉).

### plan-execute 게이트 (가치 축 #2)

복합 질문은 단순 ReAct 루프 대신 계획→실행→부족 시 재조사가 낫지만, 모든 질문에 적용하면 과함(단순 정의 조회엔 지연·비용만). **`router.py`의 `effort`가 천연 게이트**:

- `effort="high"` (다중 시스템 비교, 인과사슬, 경쟁 이론 평가) → **plan-execute** 경로
- `effort="low"|"medium"` → 단순 단발/ReAct 툴 루프

분류값이 이미 산출되므로 분기 비용은 0. plan-execute 자체는 base ReAct보다 무거우므로(별도 planner 또는 plan 주입 middleware), 이 게이트가 적용 범위를 high로 한정.

## 4. 인용 / 출처 보존 (research assistant 핵심)

이제 인용 네임스페이스가 둘:
- **책**: `[src: grossberg_ch4.pdf | §<section> | p.<page>]`
- **웹**: URL

요구사항:
- system_prompt가 두 출처를 **명시적으로 구분**. 웹 사실이 책 인용처럼 새지 않게.
- `search_knowledge` 컨텍스트 모드에서 `[src:]` 마커가 ToolMessage에 살아남아야 함 → agent가 References 재구성 가능.
- `return_answer=True` 경로에서도 인용 유지 확인 필요(LightRAG 답변에 마커 흐름).

## 5. "검색 전략을 스킬로 제공"

라우팅 규칙 + mode별 사용 지침의 패키징 형태(미결, §7):
- **Claude Code SKILL.md**: 향후 세션이 이 KB를 일관되게 구동하도록 전략을 문서·재사용 자산으로.
- **LangChain agent 구성요소**: `router.py`/`prompts/` + `system_prompt`로 런타임 코드화.
- **둘 다**: 런타임 코드 + 문서화 스킬.

## 6. 재사용 자산 (현재 코드)

| 기존 | agentic 구조에서 |
|---|---|
| `models.py` provider 라우팅 | `create_agent`의 model(gpt-5.5) 래핑 |
| `router.py` 분류기 | 프리필터 게이트(권장 A) |
| `cite.py` / `[src:]` 마커 | 툴 컨텍스트에 그대로 |
| `prompts/answer_system.md` | agent system_prompt 기반 |
| append-only cited 메모리 (`chat.py`) | `conversation_history`로 툴에 주입 |
| `tracing.py` Arize 스팬 | tool-call / RETRIEVER 스팬으로 확장 |
| `rerank.py` LLM 재랭킹 | `enable_rerank`로 툴 내부 유지 |

## 7. 미결 결정

1. **키워드 1리스트 vs hl/ll 2분리** — 권장: **2분리**(`concepts`/`entities`). 라우팅이 인자에 녹음.
2. **`return_answer` 기본값** — 권장: **`False`**(컨텍스트만). gpt-5.5가 직접 종합, 인용·멀티홉 유지. `True`는 빠른 조회용으로만.
3. **`web_search` 백엔드** — Tavily / Exa (LangChain 네이티브) vs 하네스 WebSearch. *미정.*
4. **router 처리** — (A) 프리필터 유지 vs (B) middleware 흡수. 권장: **(A)**.

## 8. 검증 계획

전환의 가치 근거는 §1 핵심 3축(웹+요약 / plan-execute / 느슨한 결합)이며, 이는 구조 자체에서 나오므로 측정에 의존하지 않는다. 따라서 아래는 ROI 게이트가 아니라 **부차적 최적화 측정**:

- **mode 라우팅 vs 항상 hybrid** (선택적): LightRAG `hybrid`가 이미 local+global을 섞으므로 mode 자동선택의 마진은 작을 수 있음. `ab_test.py` + Arize로 동일 질문셋을 두 구성(고정 hybrid vs agentic mode-routing)으로 돌려 품질·인용 정확도·지연·비용 비교. 이득이 없으면 hybrid 고정해도 3축 가치는 유지됨.
- **멀티홉/web_search 효용**: plan-execute·웹검색이 실제로 답을 개선하는 케이스 수집(특히 `effort=high` 질문).
- **키워드 주입 절감**: concepts/entities 직접 주입이 내부 키워드 추출 LLM 콜 제거로 얼마나 비용을 줄이는지 계측.
