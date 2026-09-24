# 현재 실험 기준과 BF16 변경 기록

사용자가 첨부한 복구 문서와 이후 대화에서 승인한 조건을 현재 기준으로 사용한다.
이 실험은 개인 실험이며, 아래의 이전 TinyLlama 기록은 현재 규격으로 적용하지 않는다.

- 모델: `Qwen/Qwen2.5-1.5B`
- 고정 revision: `8faed761d45a263340a0528343f099c05c9a4323`
- **실험 dtype: `torch.bfloat16` (BF16)** — 2026-09-24 사용자 승인으로 FP16에서 변경
- Attention: `eager`; RTX 3090 한 장에 모델 전체 배치; quantization/offload 없음
- `model.eval()`, 추론 시 `torch.inference_mode()`, seed 42, matmul TF32 off
- Python 3.10.12; PyTorch 2.3.1+cu121; Transformers 4.46.3; Accelerate 0.34.2;
  huggingface-hub 0.26.2; safetensors 0.4.5; NumPy 1.26.4
- 앞으로 KV Cache ON/OFF 양쪽에 동일한 BF16 조건을 적용한다.
- 첫 파일럿은 128개 고정 입력 토큰, 정확히 128개 새 토큰, batch 1, greedy generation이다.
  각 조건 warm-up 2회와 본측정 5회를 사용하며, 측정 순서는 반복마다 OFF→ON / ON→OFF로 교대한다.
- 첫 파일럿은 `model.generate()` 전체 latency, 생성 토큰 기준 throughput, peak GPU memory를 측정한다.
  Prefill/decode 분리와 확장 실험은 파일럿 성공 후 진행한다.

## 고정 입력 (9단계)

- 파일: `data/shared_5_1_input.json`
- 첨부 문서의 token IDs 128개를 순서 그대로 저장했다. 모델 ID와 revision은 위의 고정 모델과 같다.
- `tokenizer_add_special_tokens=false`; 파일럿에서는 `token_ids`로 정수 입력 tensor를 직접 만들고 문자열을 다시 tokenize하지 않는다.
- Token ID 목록의 SHA-256: `5fcc0f299f56df0dcefe9385960670e79276e2727011ef96bb0654b06186fa34`
- 해시 대상: `json.dumps(token_ids).encode('utf-8')` — 기본 JSON 직렬화의 쉼표 뒤 공백을 포함하고 끝 개행은 포함하지 않는다.
  이 값은 metadata를 포함한 JSON 파일 전체의 해시가 아니다.
- 검증: `python -B src/validate_fixed_input.py`
- 결과: `results/fixed_input_validation.json`
- 이후 코드는 `validate_fixed_input.load_fixed_input()`으로 모델 정보·토큰 수·해시를 검사한 입력을 읽을 수 있다.

## 파일럿 실행 설정 (10단계)

- 설정 파일: `docs/pilot_config.json`
- 읽기·검증 및 실행 설정 생성: `src/pilot_config.py`
- 검증 명령: `python -B src/pilot_config.py`; 결과: `results/pilot_config_validation.json`
- `load_pilot_config()`은 모델 출처, 고정 입력 해시, Python/패키지 버전과 실험 조건을 확인한다.
- `generation_config_for(config, use_cache)`는 ON/OFF에서 `use_cache`만 다른 새 `GenerationConfig`를 만든다.
- `min_new_tokens=max_new_tokens=128`, `do_sample=false`, `num_beams=1`로 길이와 greedy 조건을 고정한다.
  실제 실행 후 새 토큰 수가 128인지 별도로 검사해야 한다.
- EOS/BOS/PAD ID는 로컬 모델 설정에 맞춰 `151643`으로 고정한다. EOS 조기 종료는 `min_new_tokens`로 방지한다.
- `return_dict_in_generate`, `output_logits`, `output_scores`, `output_attentions`, `output_hidden_states`는 모두 false다.
  측정용 출력은 token ID tensor이며, 진단용 logits 등을 보관하는 비용을 포함하지 않는다.
- `configure_runtime(config)`는 Python/NumPy/PyTorch seed를 42로 맞추고 matmul TF32를 끈다.
- `model_loading_kwargs(config)`는 BF16·eager·로컬 safetensors 로딩 인자를 만든다.
  실제 실행 코드는 모델을 `cuda:0`으로 이동해 `eval()`을 적용하고 `torch.inference_mode()` 안에서 실행해야 한다.
- 모델 parameter는 BF16이고, token ID 입력은 정수형 `torch.int64`를 사용한다.
- Warm-up은 OFF/ON 각 2회, 본측정은 각 5회로 설정했다. 본측정 순서는
  `OFF→ON`, `ON→OFF`, `OFF→ON`, `ON→OFF`, `OFF→ON`이다.
- 10단계에서는 설정만 검증한다. 모델 로딩, generation, warm-up과 본측정은 실행하지 않는다.

## Warm-up (11단계)

- 실행: `python -B src/run_warmup.py`; 검증 기록: `results/warmup_validation.json`
- 고정 token IDs를 직접 GPU에 올린 후, 같은 모델에서 OFF 2회 → ON 2회 실행한다.
- 매 실행마다 새 generation config를 사용하며, 이전 실행의 KV Cache를 넘기지 않는다.
- BF16·eager·eval·inference mode를 적용하고 새 토큰 128개 및 입력 prefix 보존을 검사한다.
- 임시 forward hook으로 각 생성 단계의 다음 토큰용 raw logits 유한성을 검사한다.
  이 hook은 성공·실패 시 모두 제거되며, GPU logits나 KV Cache를 기록에 보관하지 않는다.
- 실행 사이 출력 tensor를 해제하고 GC·GPU synchronize를 수행한다. CUDA allocator cache는 비우지 않는다.
- Warm-up 기록은 성능 통계에서 제외한다. 이 단계에서는 latency·throughput·peak memory를 측정하지 않는다.
- `run_warmup(model, inputs, config)`는 이후 본측정 프로세스에서 재사용한다.
  이번 단독 실행이 다음 Python 프로세스의 warm-up을 대신하지 않으므로, 본측정 직전에 같은 모델·프로세스에서 다시 실행해야 한다.

## 본측정 반복 루프 (12단계)

- 12단계 당시 실행 검증 기록: `results/pilot_loop_validation.json` (성능 측정 전 기록으로 보존).
- `src/run_pilot.py`의 현재 실행 범위와 출력 파일은 아래 15단계를 따른다.
- 모델과 GPU 입력을 한 번 준비하고, 같은 프로세스에서 warm-up OFF 2회 → ON 2회를 실행한다.
- Warm-up 진단 hook이 제거된 것을 확인한 뒤 아래 순서로 각 조건 5회씩 실행한다.
  - repeat 0: OFF → ON
  - repeat 1: ON → OFF
  - repeat 2: OFF → ON
  - repeat 3: ON → OFF
  - repeat 4: OFF → ON
- 각 실행은 새 generation config를 사용한다. 고정 입력에서 다시 시작하며 이전 KV Cache를 전달하지 않는다.
- 실행 사이 출력 GPU tensor를 해제하고 GC·synchronize를 수행한다. 기록에는 CPU token IDs만 보관한다.
- 각 실행에서 새 토큰 수 128개, 입력 prefix 보존 및 입력 IDs·mask의 불변성을 검사한다.
- Warm-up과 반복 실행 기록은 별도 배열에 저장한다. 실패하면 중단하고 부분 결과와 오류를 남긴다.
- 12단계는 반복 루프의 실제 실행 검증이다. Latency·throughput·peak memory와 성능 통계는 아직 없으며,
  이번 기록은 성능 측정 표본에 포함하지 않는다. 측정 기능은 13–15단계에서 추가한다.

## Total latency (13단계)

- 13단계 측정 기록: `results/pilot_latency.json` (보존). 현재 실행 명령과 출력은 아래 15단계를 따른다.
- 같은 프로세스에서 모델·GPU 입력 준비 → warm-up 4회 → 교대 본측정 10회를 실행한다.
- 매 실행마다 generation config 준비와 GC를 마친 뒤, inference mode 안에서 아래 순서로 측정한다.
  1. `torch.cuda.synchronize(device)`
  2. `started = time.perf_counter()`
  3. `outputs = model.generate(...)`
  4. `torch.cuda.synchronize(device)`
  5. `total_latency_seconds = time.perf_counter() - started`
- 측정값은 초 단위이며 prefill과 decode를 포함한 전체 generation의 경과 시간이다.
  GPU kernel 시간만이 아니라 `generate()` 내부 CPU 작업과 완료 동기화 비용도 포함한다.
- 모델 로딩, 입력 GPU 전송, config 생성, GC, warm-up, 출력 검증·CPU 복사 및 결과 저장은 측정 구간 밖이다.
- Warm-up 진단 hook을 제거한 상태로 측정한다. 매 실행에서 새 토큰 128개 및 유한한 양수 latency를 확인한다.
- 성공한 본측정 기록만 `included_in_statistics=true`로 표시하며 warm-up은 계속 제외한다.
  현재 단계에서는 raw latency 5회씩만 저장하고 통계를 계산하지 않는다.
- Throughput, peak memory와 최종 mean/std·speedup·ON/OFF 출력 일치 검증은 후속 단계에서 추가한다.

## Throughput (14단계)

- 정의: `throughput_tokens_per_second = actual_generated_tokens / total_latency_seconds`.
- 분자는 출력에서 확인한 실제 새 토큰 수이며 prompt 토큰은 포함하지 않는다.
  분모는 prefill과 decode를 포함하는 전체 generation latency다. Decode 전용 throughput은 아니다.
- `src/calculate_throughput.py`의 공통 함수가 양수 토큰 수·유한한 양수 latency를 검사하고 계산한다.
- 기존 측정값 사용: `python -B src/calculate_throughput.py`.
  13단계 `results/pilot_latency.json`을 읽고 GPU 재실행 없이 `results/pilot_throughput.json`에 저장한다.
  원본 latency·출력 IDs·측정 시각·측정 코드 해시는 그대로 보존하고, 계산 출처와 시각은 `derivation`에 별도로 기록한다.
- 새로 측정하는 `src/run_pilot.py`도 동일한 계산 함수를 사용한다.
  현재는 아래 15단계에 따라 메모리까지 함께 측정하고 `results/pilot_memory.json`에 저장한다.
  14단계 기존 결과는 별도로 보존한다.
- Throughput 계산은 timer 종료 후 수행한다. Warm-up에는 throughput을 계산하지 않으며 통계에서도 제외한다.
- 이 단계에서는 기존 10개 표본에 대한 throughput만 계산·검증한다.
  Peak memory와 최종 mean/std·speedup·ON/OFF 출력 일치 검증은 후속 단계에서 진행한다.

## GPU memory (15단계)

- 실행: `python -B src/run_pilot.py`; 결과: `results/pilot_memory.json`.
- 같은 모델·프로세스에서 warm-up 4회 후, 고정 교대 순서로 OFF/ON 각 5회를 측정한다.
- 매 실행 순서: 이전 출력 해제 → `gc.collect()` → `torch.cuda.synchronize(device)` →
  `memory_allocated()` / `memory_reserved()` baseline 기록 → `reset_peak_memory_stats()` → generation.
- Generation 완료 동기화와 timer 종료 직후 `max_memory_allocated()` 및 `max_memory_reserved()`를 읽는다.
  출력 길이 검증 등 후속 GPU 연산이 peak에 섞이지 않도록 peak를 먼저 읽는다.
- Baseline·peak 조회 및 peak 초기화는 latency 구간 밖이다. Latency와 throughput도 같은 실행에서 기록한다.
- 원시 값은 bytes이며 화면 출력만 MiB (`bytes / 2**20`)로 변환한다.
- Baseline은 상주 모델·입력을 포함한다. `peak_allocated_above_baseline_bytes`는 allocated peak에서
  baseline을 뺀 값이며 임시 tensor 등을 포함하므로 KV Cache 자체 크기로 해석하지 않는다.
- Allocated는 PyTorch가 tensor 등에 할당 중인 메모리, reserved는 재사용 가능한 allocator cache까지 포함한다.
  매 실행마다 `empty_cache()`를 호출하지 않으므로 reserved에는 warm-up과 앞선 실행의 영향이 남을 수 있다.
- 이 값은 해당 프로세스의 PyTorch CUDA allocator 통계다. GPU 전체 사용량이나 PyTorch 외부 할당량은 아니다.
- Warm-up은 성능 통계에서 제외한다. 이번 단계는 10개 raw 표본 저장까지 수행하며,
  최종 mean/std·speedup·ON/OFF 출력 일치 검증은 16단계에서 진행한다.

## 결과 검증·통계·저장 (16단계)

- 실행: `python -B src/summarize_pilot.py`. GPU 재실행이나 패키지 설치 없이 저장된 15단계 표본만 분석한다.
- Raw JSON: `results/pilot_memory.json` (원본 보존); raw CSV: `results/pilot_raw.csv` (본측정 10행).
- 검증과 집계 결과: `results/pilot_summary.json`. 측정 시각과 분석 시각, raw 파일 및 코드 해시를 구분해 기록한다.
- 고정 입력·설정·측정 코드 해시, warm-up 제외, 교대 순서, 각 조건 5회, 실제 새 토큰 128개,
  출력 token ID 해시, throughput 계산 및 메모리 baseline/peak/delta의 정합성을 검증한다.
- ON/OFF는 해시만 비교하지 않고 실제 token ID 배열을 비교한다. 같은 repeat의 5쌍과 전체 10회를
  첫 OFF 출력에 대조하며 첫 유효 OFF/ON token IDs도 summary에 보존한다.
- 불일치 시 최초 차이의 새 토큰 기준 0-based 위치와 양쪽 ID를 저장하고 실패로 종료한다.
  손상된 hash·누락된 실행·유효하지 않은 측정값은 집계 전에 오류로 처리한다.
- 각 조건의 표본 수는 5이며 표준편차는 표본 표준편차 (`ddof=1`)다. 이상치를 제거하지 않는다.
- Throughput 평균은 실행별 `actual_generated_tokens / latency`의 산술평균이다.
- Speedup은 `mean_OFF_latency / mean_ON_latency`이며 1보다 커야 ON이 더 빠르다는 뜻이다.
- 메모리 통계의 원시 단위는 bytes, 표시 단위 MiB는 `bytes / 2**20`이다.

이번 표본은 전체 10회 token IDs가 일치했다. 평균 ± 표본 표준편차는 다음과 같다.

| 지표 | OFF | ON |
|---|---:|---:|
| Total latency (s) | 3.227592 ± 0.004935 | 3.252939 ± 0.038871 |
| Throughput (tokens/s) | 39.658127 ± 0.060574 | 39.353463 ± 0.463296 |
| Peak allocated (MiB) | 2970.900879 ± 0 | 2965.484375 ± 0 |
| Peak reserved (MiB) | 3184 ± 0 | 3184 ± 0 |

Speedup은 **0.992208×**로, 이번 128-input/128-generation 조건에서는 속도 향상이 관측되지 않았다.
이는 각 조건 5회에 대한 기술 통계이며, 성능 차이의 통계적 유의성이나 다른 길이에서의 성능을 주장하지 않는다.
Peak allocated 차이를 KV Cache 자체 크기로 해석하지 않는다.
파일럿 그래프와 Git 백업 범위는 아래 17단계, 첫 길이 확장 조건은 18단계를 참고한다.

## 그래프와 Git 백업 (17단계)

- 그래프 생성: `python -B src/plot_pilot.py`. 저장된 raw JSON과 summary의 출처 해시를 확인하고,
  모델 실행 없이 `plots/pilot_results.png`와 `plots/pilot_results.pdf`를 생성한다.
- 설치된 Matplotlib을 사용하며 새 패키지는 설치하지 않는다. 버전·입력/출력 해시는 `results/plot_metadata.json`에 기록한다.
- Latency, throughput, peak allocated, baseline 대비 allocated 증가량을 표시한다.
  점은 개별 표본 5개, 다이아몬드와 error bar는 평균 ± 표본 표준편차 (`ddof=1`)다.
  모든 y축은 0에서 시작하며 공통 baseline·peak reserved 및 speedup도 명시한다.
- README에 결과 요약, 그래프, 원본 데이터와 검증 결과 링크, 실행 명령을 정리한다.
- 백업 범위: 실험 코드·규격·환경 기록·고정 입력·측정/검증 CSV·JSON·PNG/PDF.
  모델 가중치·cache·가상환경은 `.gitignore`로 제외한다.
- 원격 백업은 `origin/main`으로 push가 성공하고 remote commit과 로컬 HEAD가 일치할 때 완료된다.

## Prompt length sweep (18단계, 첫 확장)

- 사용자 승인으로 generation=128을 유지한 prompt 128·256·512·1024 sweep을 진행한다.
- 세부 조건·입력 구성·실행 명령·해석 범위는 [prompt_sweep.md](prompt_sweep.md)를 따른다.
- 기존 고정 IDs를 반복한 길이별 입력과 해시를 별도 파일에 저장한다.
- 같은 모델·프로세스에서 길이마다 warm-up 4회, 본측정 10회로 총 40개 표본을 얻는다.
- 파일럿과 별도의 raw CSV/JSON·summary·그래프를 생성한다.
- Prefill/decode 분리나 generation length sweep은 이번 확장에 포함하지 않는다.

## Generation length sweep (18단계, 두 번째 확장)

- 입력 128개를 유지하고 생성 길이 32·64·128·256을 비교한다.
- 별도 설정의 min/max new tokens를 해당 길이로 고정하며, 각 조건 warm-up 2회·본측정 5회를 유지한다.
- 세부 조건·시간 측정 경계·메모리 정의·출력 검증·재현 명령은 [generation_sweep.md](generation_sweep.md)를 따른다.
- 이전 파일럿과 prompt sweep 결과는 보존하며, generation sweep 전용 CSV·JSON·PNG/PDF를 저장한다.
- Prefill/decode 분리 및 최종 KV tensor 크기 sweep은 아직 수행하지 않는다.

## 두 sweep의 6개 지표 확장

- 사용자 요청으로 prefill/decode/total latency, decode throughput, peak allocated, actual KV tensor size를 측정한다.
- 동일한 수동 greedy 루프로 두 sweep을 새로 실행하며 기존 generate 전체 시간 표본은 보존한다.
- Prefill에는 첫 token 생성, decode에는 나머지 N−1개 token을 포함한다. 실제 KV는 모든 layer의 K/V tensor를 합산한다.
- 측정 경계·입력 구성·기존 출력과의 대조·재현 방법은 [detailed_sweeps.md](detailed_sweeps.md)를 기준으로 한다.
- 앞선 단계의 “분리 측정 미수행”은 해당 단계 당시의 상태이며, 이번 확장부터 별도 파일로 측정한다.

## 변경 근거와 검증 기록

FP16 + eager에서 첫 attention의 scaling 이전 Q×K 값이 약 215,584로 관측되어,
FP16 최대 유한값 65,504를 초과했다. 최종 logits가 NaN이 되었고 생성 결과가 token ID 0으로 반복됐다.
BF16 + eager 대조 실행에서는 정확히 128개 새 토큰이 생성되고 모든 생성 단계의 raw logits가 유한했다.
패키지나 attention 구현은 바꾸지 않고 실험 dtype을 BF16으로 변경한다.

- [FP16 실패 및 BF16 대조 진단](../results/generation_diagnostic.json)
- 현재 로딩 검증: `src/validate_model_load.py` → `results/model_load.json`
- 현재 generation 검증: `src/validate_generation.py` → `results/basic_generation.json`
- 기본 generation/KV Cache 동작 검증은 임시 입력을 사용한다. 파일럿은 위의 고정 입력만 사용한다.
- `output_logits=True` 및 매 단계 유한성 검사는 기능 검증용이다. 성능 측정에는 이 저장·검사 비용을 포함하지 않는다.
- 원래 FP16에서 실패한 generation 결과는 진단 JSON 안의 `fp16_generation_validation`에 보존한다.

---

## 이전 실험 기록 — 현재 규격으로 적용하지 않음

나는 NLP 기초연구 토이 프로젝트로 다음 프로젝트를 진행하고 있다.

[프로젝트 주제]
KV Cache 적용에 따른 LLM 추론 효율 및 메모리 사용 특성 분석

이 프로젝트에서는 autoregressive LLM inference에서 KV Cache ON/OFF를 비교하여 다음 네 가지 지표를 측정하려고 한다.

1. Total inference latency
2. Decode latency
3. Token generation throughput (tokens/sec)
4. Peak GPU memory usage

핵심 목적은 단순히 "KV Cache가 빠르다"를 확인하는 것이 아니라,

KV Cache ON
→ 이전 token의 K/V 재계산 감소
→ decode latency 감소
→ throughput 증가

동시에

KV Cache ON
→ 이전 token의 K/V를 GPU memory에 저장
→ sequence length 증가에 따라 memory usage 증가

라는 compute-memory trade-off를 실제 GPU에서 정량적으로 확인하는 것이다.


==================================================
1. 현재 환경
==================================================

VESSL Workspace를 사용하고 있고 VSCode Remote-SSH 연결까지 완료되어 있다.

Workspace:
- OS: Linux
- GPU: NVIDIA GeForce RTX 3090 × 1
- VRAM: 24 GB
- CPU: 15
- RAM: 60 GiB

확인된 소프트웨어 환경:
- Python 3.10.12
- PyTorch 2.3.1+cu121
- CUDA Toolkit 12.1
- NVIDIA Driver 550.107.02
- torch.cuda.is_available() == True
- GPU tensor 연산 테스트 성공
- cuDNN 8.9.2

설치한 패키지:
- transformers==4.44.2
- accelerate==0.34.2
- huggingface_hub==0.24.6
- safetensors==0.4.5

기존 PyTorch/CUDA 환경은 정상적으로 동작하고 있으므로,
가급적 torch나 CUDA 관련 패키지를 다시 설치하거나 업그레이드하지 말 것.


==================================================
2. 프로젝트 폴더
==================================================

현재 프로젝트 경로:

/root/kv-cache-project

현재 구조:

kv-cache-project/
├── src/
├── results/
└── plots/

현재 src 폴더에는 대략 다음 파일들이 있다.

src/test_model_load.py
- Hugging Face model loading 테스트

src/test_generation.py
- 실제 autoregressive text generation 테스트

src/test_kv_cache.py
- use_cache=True/False일 때 past_key_values가 실제 생성되는지 확인


==================================================
3. 현재 사용 중인 모델
==================================================

현재 모델:

TinyLlama/TinyLlama-1.1B-Chat-v1.0

설정:
- 약 1.1B parameters
- torch.float16
- CUDA GPU
- model.eval()
- torch.inference_mode()

모델 loading은 정상적으로 성공했다.

Parameters:
1100048384

실제 generation도 정상 동작했다.

Chat model이므로 tokenizer.apply_chat_template()를 사용했으며,
50개의 new token 생성도 정상적으로 확인했다.


==================================================
4. KV Cache 동작 확인 결과
==================================================

use_cache=True와 use_cache=False를 직접 model forward에서 확인했다.

Input length 6 token 기준:

KV Cache ON:

past_key_values is None: False
Number of layers: 22

First layer K:
torch.Size([1, 4, 6, 64])

First layer V:
torch.Size([1, 4, 6, 64])

KV Cache OFF:

past_key_values is None: True

따라서 use_cache=True일 때 실제로 past_key_values가 생성되고,
use_cache=False일 때는 생성되지 않는 것을 확인했다.

TinyLlama는 KV head가 4개인 구조이므로 GQA 계열 모델이라는 점도 확인할 수 있다.

Transformers에서 tuple past_key_values deprecated warning이 나오지만,
현재 실험에는 영향을 주지 않는다.


==================================================
5. 최종 기본 실험 설계
==================================================

프로젝트 전체 기본 실험은 다음과 같다.

[실험 1: Prompt Length 변화]

Prompt Length:
128
256
512
1024 tokens

Generation Length:
128 tokens 고정

각 조건에서:
- KV Cache ON
- KV Cache OFF

비교


[실험 2: Generation Length 변화]

Prompt Length:
128 tokens 고정

Generation Length:
32
64
128
256 tokens

각 조건에서:
- KV Cache ON
- KV Cache OFF

비교


중요:
prompt length는 문자열 길이가 아니라
tokenizer 기준 실제 token 수를 정확히 맞춰야 한다.


==================================================
6. 이번 주 목표
==================================================

이번 주에는 전체 실험 sweep을 완성하는 것보다 우선

"KV Cache ON/OFF 조건에서 네 가지 지표가 정확하게 측정되는 benchmark 코드"

를 완성하는 것이 목표다.

이번 주 반드시 측정하고 싶은 지표:

1. Total inference latency
2. Decode latency
3. Token generation throughput (tokens/sec)
4. Peak GPU memory usage

먼저 하나의 대표 조건에서 측정 코드가 정상적으로 동작하는 것을 확인하고 싶다.

예:
Prompt Length = 128
Generation Length = 128

KV Cache ON / OFF

이 조건에서 네 지표가 정확하게 측정되는지 검증한다.

정상 동작하면 이후 여러 prompt/generation length로 확장할 예정이다.


==================================================
7. 구현에서 중요한 원칙
==================================================

처음부터 vLLM, TensorRT-LLM 같은 복잡한 inference framework는 사용하지 않는다.

우선:

PyTorch
+
Hugging Face Transformers

만 사용한다.

또한 가능하면 model.generate() 내부 동작만 믿지 말고,
autoregressive decode loop를 직접 구현해서
KV Cache ON/OFF의 차이를 명확하게 제어하고 싶다.


KV Cache ON의 이상적인 동작:

Prefill:
전체 prompt를 한 번 입력
→ past_key_values 생성

Decode:
매 step마다 새 token 1개만 model에 입력
+
past_key_values 재사용


KV Cache OFF:

매 decode step마다

prompt
+
지금까지 생성된 모든 token

전체 sequence를 다시 model에 넣는다.

즉 ON/OFF의 실제 계산 구조 차이가 코드에 명확하게 드러나야 한다.


==================================================
8. Latency 측정 시 주의사항
==================================================

CUDA는 asynchronous execution이므로 단순히

start = time.time()
model(...)
end = time.time()

같이 측정하면 안 된다.

최소한 다음과 같이 synchronization을 고려해야 한다.

torch.cuda.synchronize()
start = time.perf_counter()

# GPU inference

torch.cuda.synchronize()
end = time.perf_counter()

또는 CUDA Event를 사용하는 방법도 고려할 수 있다.

측정 방법을 선택할 때 왜 그 방법을 쓰는지 설명해줘.


Total inference latency와 Decode latency를 가능하면 구분하고 싶다.

개념적으로:

Total latency
≈ Prefill latency + Decode latency

형태로 볼 수 있도록 구현하고 싶다.


==================================================
9. Throughput 정의
==================================================

Throughput은 generated token 기준으로 계산하고 싶다.

tokens/sec =
생성한 new token 수 / decode time

또는 필요하면 total latency 기준 throughput도 같이 기록할 수 있지만,
주요 지표는 decode throughput이다.

Prompt token은 throughput numerator에 포함하지 않는다.


==================================================
10. GPU Memory 측정
==================================================

Peak GPU memory는 PyTorch CUDA memory API를 사용하고 싶다.

예를 들어:

torch.cuda.reset_peak_memory_stats()

실험 실행

torch.cuda.max_memory_allocated()

등을 이용해서 측정하고 싶다.

하지만 모델 weight 자체가 차지하는 baseline memory와
inference 중 추가되는 memory를 어떻게 구분하는 것이 좋은지도 설명해줘.

필요하면 다음 두 값을 모두 저장해도 된다.

- absolute peak GPU memory
- model load 이후 추가 peak memory

중요한 것은 KV Cache ON/OFF를 공정하게 비교하는 것이다.


==================================================
11. 실험 공정성을 위해 통제할 조건
==================================================

KV Cache 외에는 가능한 한 동일하게 유지해야 한다.

예:

- 동일한 model
- 동일한 tokenizer
- 동일한 prompt token IDs
- 동일한 generation length
- batch size = 1
- torch.float16
- 동일한 device
- model.eval()
- torch.inference_mode()
- greedy decoding
- do_sample=False
- 동일한 EOS 처리 방법
- 동일한 random seed
- 동일한 warm-up 횟수
- 동일한 반복 측정 횟수

가능하면 generation 중 EOS가 조기에 나와서 실제 생성 token 수가 달라지는 문제를 피하고 싶다.

benchmark에서는 정확히 지정된 token 수만큼 decode하도록 설계하는 방법도 고려해줘.


==================================================
12. Warm-up 및 반복 측정
==================================================

실제 benchmark 전에 여러 번 warm-up inference를 수행하고 싶다.

그 후 같은 조건을 여러 번 반복하여

- mean
- 필요하면 std

를 계산한다.

아직 반복 횟수는 정하지 않았으므로
너무 많지 않으면서 신뢰성 있는 값을 얻을 수 있는 횟수를 제안해줘.


==================================================
13. 앞으로 저장할 결과
==================================================

나중에는 CSV로 다음과 같은 데이터를 저장할 예정이다.

예:

model
kv_cache
prompt_length
generation_length
prefill_latency_ms
decode_latency_ms
total_latency_ms
throughput_tokens_per_sec
peak_memory_mb
additional_peak_memory_mb

필요한 column이 더 있다면 제안해줘.


==================================================
14. 향후 확장 실험
==================================================

기본 KV Cache ON/OFF 실험을 모두 완료한 뒤 시간이 남으면

MHA
vs
GQA

또는 KV head 수가 다른 모델을 비교하여

- KV Cache size
- GPU memory usage
- latency
- throughput

차이를 분석할 예정이다.

하지만 지금은 확장 실험을 하지 않는다.


==================================================
15. 내가 읽은 관련 논문
==================================================

다음 논문을 읽은 상태다.

- Attention Is All You Need
- Fast Transformer Decoding: One Write-Head is All You Need
- GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

따라서 다음 개념은 어느 정도 알고 있다.

- Transformer self-attention
- autoregressive inference
- prefill / decode
- KV Cache
- MHA
- MQA
- GQA
- incremental decoding
- KV tensor memory bandwidth bottleneck


==================================================
16. 작업 방식
==================================================

나는 LLM inference benchmark 구현이 처음이다.

따라서 전체 프로젝트 코드를 한 번에 작성하지 말고,
한 단계씩 진행해줘.

현재 다음 단계는:

"manual autoregressive decoding으로 KV Cache ON/OFF를 구현하고,
같은 token이 생성되는지 확인한 뒤,
latency / throughput / GPU memory 측정을 추가"

이다.

먼저 현재 프로젝트의 파일을 확인해줘.

중요:

1. 먼저 기존 파일을 읽고 현재 구현 상태를 파악할 것.
2. 처음부터 프로젝트 전체를 다시 만들지 말 것.
3. 기존 PyTorch/CUDA 환경을 불필요하게 수정하지 말 것.
4. 파일을 수정하기 전에 어떤 파일을 왜 수정하려는지 설명할 것.
5. 한 번에 너무 많은 코드를 추가하지 말 것.
6. 각 단계가 성공한 것을 확인한 뒤 다음 단계로 넘어갈 것.
7. benchmark의 정확성이 가장 중요하며, 단순히 코드가 실행되는 것만으로 끝내지 말 것.

우선 현재 /root/kv-cache-project 내용을 읽고,
지금까지 구현된 파일들이 무엇을 하는지 요약해줘.

그 다음 이번 주 목표를 위해
"다음 한 단계"만 제안해줘.

아직 코드를 수정하지는 말아줘.