# Grossberg Chapter 4 — 주요 Figure 및 모델 목록

## 모델 계보

Chapter 4의 모델들은 병렬 관계가 아니라 **점진적 확장(embedding)** 관계로 이어진다.
Grossberg는 이를 "Embedding Principle"이라 부른다: 선행 모델이 후속 모델 안에 완전히 포함된다.

```
BCS / FCS  (2D, 비층판)
│  └─ Double filter + grouping network  [Fig 4.32]
│  └─ FIDO (Filling-In DOmain)
│  └─ Grossberg-Todorovic model  (정량 시뮬레이션, 1988)
│
├─→  FACADE  (3D 확장, 비층판, 1994/1997)
│      BCS/FCS의 경계·표면 메커니즘을 3D 깊이 지각으로 확장
│
LAMINART  (층판 구현, 2D, 1999)
│      BCS/FCS를 V1/V2 층판 피질 회로로 구체화
│      ART의 top-down 기대·주의 회로 통합
│
└─→  3D LAMINART  (3D + 층판, 1990s~현재)
       LAMINART의 3D 확장 = FACADE의 층판 구현
       현재까지 가장 광범위한 설명·예측 범위를 가진 시각 피질 이론

비교 모델
└─  Complex channels model  (Sutter, Beck, Graham, 1989)
       텍스처 분리 모델 — BCS와 비교 대조되며 BCS의 우위를 보여줌 [Fig 4.34]

의식 관련
└─  Surface-shroud resonance
       V4-PPC 간 공명으로 의식적 시각 지각을 트리거
```

---

## 모델별 핵심 역할

| 모델 | 핵심 역할 |
|------|-----------|
| **BCS** (Boundary Contour System) | 경계 형성: simple/complex/hypercomplex cell + bipole cell 그룹핑 |
| **FCS** (Feature Contour System) | 표면 채워넣기: illuminant 보정 후 색/밝기를 FIDO 내에서 확산 |
| **Grossberg-Todorovic model** | BCS/FCS 기반 2D 밝기 filling-in 정량 시뮬레이션 (1988) |
| **Double filter + grouping network** | Simple→Complex→Hypercomplex→Bipole의 2단 필터 구조 [Fig 4.32] |
| **FIDO** (Filling-In DOmain) | ON/OFF 표면 채워넣기 도메인 — FCS의 신경 구현체 |
| **FACADE** (Form And Color And DEpth) | BCS/FCS를 3D로 확장한 비층판 시각 지각 이론 (1994/1997) |
| **LAMINART** | BCS/FCS를 V1/V2 층판 피질 회로로 구현 + ART top-down 회로 통합 (1999) |
| **3D LAMINART** | LAMINART의 3D 확장 = FACADE의 층판 구현 — 현존 최고 설명 범위 |
| **Surface-shroud resonance** | V4-PPC 간 공명으로 의식적 지각 트리거 |
| **Complex channels model** | 비교 대조용 텍스처 분리 모델 (Sutter et al.) |

---

## Figure 전체 목록 (4.1 ~ 4.58)

### 1부 — 밝기 항등성 / Filling-in (pp. 1–14)

> FCS 및 Grossberg-Todorovic model의 기초 현상들

| Fig | 주제 |
|-----|------|
| 4.1 | 경계가 filling-in의 장벽이 되는 고전적 예시 |
| 4.2 | Mach band — 동일한 밝기 차이가 다른 percept를 유발 |
| 4.3 | McCann Mondrian — illuminant 보정(brightness constancy) |
| 4.4 | 조명 기울기 하에서 인접 패치 간 밝기 점프 |
| 4.5 | Multiple-scale balanced competition으로 color contour 선택 |
| 4.6 | Color contour의 filling-in으로 illuminant-discounted 표면 복원 |
| 4.7 | 균일 조명 하 밝기 항등성 시뮬레이션 |
| 4.8 | 조명 기울기 하 밝기 항등성 시뮬레이션 |
| 4.9 | 밝기 대비(brightness contrast) 시뮬레이션 |
| 4.10 | 밝기 동화(brightness assimilation) 시뮬레이션 |
| 4.11 | Double step 및 Craik-O'Brien-Cornsweet(COCE) 착시 시뮬레이션 |
| 4.12 | 2D COCE 시뮬레이션 |
| 4.13 | Contrast constancy — 조명 기울기 하에서도 상대 밝기 역전 |
| 4.14 | Paradiso & Nakayama의 filling-in "현장 포착" 실험 (Arrington 시뮬레이션) |

### 2부 — 경계 형성 / End-cut / BCS 회로 (pp. 15–39)

> Simple cell → Complex cell → Hypercomplex cell → Bipole cell 계층 및 Double filter 구조

| Fig | 주제 |
|-----|------|
| 4.15 | Simple cell — oriented local contrast detector (edge detector 아님) |
| 4.16 | Odd simple cell 수용장 및 발화 임계값의 최소 구현 |
| 4.17 | Complex cell — 반대 극성 simple cell 통합, full-wave rectification |
| 4.18 | 단일 물체에 대한 두 망막 이미지의 시차(disparity) |
| 4.19 | V1 layer 3B에서 binocular disparity를 계산하는 층판 피질 회로 |
| 4.20 | Glass pattern vs. reverse-contrast Glass pattern의 경계 그룹핑 차이 |
| 4.21 | Simple cell이 굵은 bar 끝에서는 반응하지만 얇은 선 끝에서는 미반응 |
| 4.22 | 얇은 선 끝에서 simple/complex cell 반응 컴퓨터 시뮬레이션 |
| 4.23 | End gap이 end cut으로 닫히지 않으면 모든 선 끝에서 색이 새어나옴 |
| 4.24 | End cut 생성 — 세포 패턴 활성화에 민감해야 하는 이유 |
| 4.25 | Simple→Complex→Hypercomplex 네트워크가 end cut을 생성하는 과정 |
| 4.26 | Neon color spreading에서 end cut 형성 과정 |
| 4.27 | Bipole cell이 end cut 사이의 경계를 보간하고 최종 선택하는 과정 |
| 4.28 | Bipole cell의 두 branch(pole) 수용장 — 장거리 경계 완성 |
| 4.29 | V2에서 bipole cell의 신경생리학적 증거 (von der Heydt et al., 1984) |
| 4.30 | V1 내 장거리 수평 연결의 해부학적 증거 (Fitzpatrick lab) |
| 4.31 | 예측된 bipole cell 수용장 — 신경생리·심리물리 데이터 및 후속 모델들 비교 |
| 4.32 | **Double filter + grouping network** 전체 회로도 |
| 4.33 | 삼분·이분 텍스처 — emergent boundary grouping이 영역 분리 |
| 4.34 | Complex channels model이 틀리는 텍스처 (g, i) — BCS가 정확히 시뮬레이션 |
| 4.35 | Spatial impenetrability — pac-man 그룹핑의 허용/차단 |
| 4.36 | Banksy 그라피티 — amodal 경계 완성과 spatial impenetrability 이용 |
| 4.37 | Collinear vs. perpendicular Kanizsa square — bipole cell 특성 확인 |
| 4.38 | Bipole 장거리 협력 + hypercomplex 단거리 경쟁의 합작으로 경계 생성 |
| 4.39 | **LAMINART 모델** 개략도 — 층판 피질 해부학 및 동역학 |

### 3부 — FACADE / 3D 깊이 지각 / 표면 채워넣기 (pp. 40–65)

> FACADE macrocircuit, DaVinci stereopsis, figure-ground 분리, 3D LAMINART 투명성 설명

| Fig | 주제 |
|-----|------|
| 4.40 | Koffka-Benussi ring |
| 4.41 | Kanizsa-Minguzzi ring |
| 4.42 | Kanizsa-Minguzzi ring percept 컴퓨터 시뮬레이션 |
| 4.43 | (a) Bipole cell의 end cut 유발, (b) Necker cube 양안정 지각, (c) 공간 주의와 깊이/밝기 |
| 4.44 | **FACADE macrocircuit** — LGN에서 V1·V2·V4까지 경계·표면 형성 단계 |
| 4.45 | ON/OFF feature contour 활성이 filling-in 표면 영역을 만드는 방식 |
| 4.46 | Feature contour 입력이 경계에 인접·collinear할 때 filling-in 발생 |
| 4.47 | ON/OFF FIDO 출력 신호를 처리하는 double-opponent network |
| 4.48 | 닫힌 경계는 filling-in 억제, 열린 경계는 색 양쪽으로 확산 |
| 4.49 | DaVinci stereopsis — 왼쪽 눈이 더 많이 보는 벽 영역 |
| 4.50 | 양안·단안 경계의 합산으로 특정 깊이에서만 닫힌 3D 경계 형성 |
| 4.51 | 경계-표면 보완적 일관성 회로 — 자동으로 surface contour 억제 유도 |
| 4.52 | 3D LAMINART 모델이 random dot stereogram의 두 단안 이미지를 변환 |
| 4.53 | On-center off-surround 네트워크 — 밝은 Kanizsa square가 더 가깝게 보이는 이유 |
| 4.54 | Figure-ground 분리의 초기 단계 |
| 4.55 | V2에서의 amodal 경계·표면 완성 |
| 4.56 | V4에서 비가려진 물체의 visible, figure-ground separated 3D 표면 표현 최종 단계 |
| 4.57 | Unimodal/bistable 투명성 및 평면 2D 표면의 percept 예시 |
| 4.58 | 투명성 percept를 설명하는 **LAMINART 처리 단계** 회로도 |

---

총 **58개 figure** / 핵심 모델 **10개** (계보 관계 포함)
