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