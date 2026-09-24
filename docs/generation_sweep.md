# Generation length sweep

입력을 128 token IDs로 고정하고 생성 길이를 32·64·128·256으로 변경한다.
파일럿 및 prompt length sweep 코드는 수정하지 않으며 결과도 별도로 보존한다.

## 조건과 측정

- 모델: `Qwen/Qwen2.5-1.5B`, revision `8faed761d45a263340a0528343f099c05c9a4323`.
- RTX 3090, BF16, eager, batch 1, greedy, eval/inference mode, seed 42, matmul TF32 off.
- 입력: 기존 `data/shared_5_1_input.json`의 token IDs 128개, 재-tokenization 없음.
- 설정: `docs/generation_sweep_config.json`. 기존 설정·고정 입력 파일 SHA-256을 기록한다.
- 생성 길이 N별로 `min_new_tokens=max_new_tokens=N`만 변경한다. EOS 조기 종료를 방지한다.
- 생성 길이 32→64→128→256 순으로 실행한다. 각 길이 내 ON/OFF 차이는 `use_cache`뿐이다.
- 모델은 한 번 로딩하며 각 길이에서 OFF 2회 → ON 2회 warm-up 후, 교대 순서로 각 조건 5회 측정한다.
- 총 warm-up 16회와 본측정 40회. 이전 파일럿 및 prompt sweep의 표본을 통계에 합치지 않는다.

측정 구간은 GPU 입력을 미리 준비하고 `CUDA synchronize → perf_counter → generate → CUDA synchronize → perf_counter`이다.
출력 검증·CPU 복사·결과 저장 및 메모리 통계 조회는 시간 측정 밖이다.
Throughput은 **실제 생성 토큰 수 N / 전체 generation latency**이며 prefill을 포함한다.
기존 이미지의 decode 전용 throughput과 정의가 다르다. Prefill/decode 분리 측정은 수행하지 않는다.

매 실행 전 GC·동기화 후 allocated/reserved baseline을 기록하고 peak를 초기화한다.
Generation 완료 후 출력 검증 전에 allocated/reserved peak를 읽는다.
Allocator cache는 실행 및 길이 사이에 비우지 않으므로 reserved에 이전 실행의 영향이 남을 수 있다.
Peak allocated와 baseline 차이는 임시 tensor도 포함하며 KV Cache 자체 크기가 아니다.
최종 KV tensor 크기의 길이별 측정도 이번 실험에는 포함하지 않는다.

Warm-up의 각 forward에서 raw logits 유한성, 입력 처리 길이, `use_cache` 및 실제 cache 반환을 검증한다.
진단 hook은 본측정 전에 제거한다. 매 실행 후 생성 토큰 수가 정확히 N인지, 입력 prefix가 보존됐는지 확인한다.
길이별 ON/OFF 전체 10회 token IDs를 직접 비교하고 불일치가 있다면 첫 차이를 기록한다.
표준편차는 표본 표준편차 (`ddof=1`), speedup은 OFF 평균 latency / ON 평균 latency다.
이상치를 제거하지 않는다. 오름차순 단일 sweep이므로 시간 경과·클럭 변화의 영향을 완전히 분리하지 않는다.

## 실행 및 산출물

```bash
python -B src/run_generation_sweep.py
python -B src/summarize_generation_sweep.py
python -B src/plot_generation_sweep.py
```

- Raw JSON: `results/generation_sweep_raw.json` (warm-up과 본측정 분리).
- Raw CSV: `results/generation_sweep_raw.csv` (본측정 40행).
- 통계·검증: `results/generation_sweep_summary.json`.
- 그래프: `plots/generation_sweep.png`, `plots/generation_sweep.pdf`.
- 그래프 출처: `results/generation_sweep_plot_metadata.json`.

각 명령은 generation sweep 전용 결과 파일을 갱신한다. 패키지 설치와 모델 다운로드는 필요 없다.

## 실행 결과

각 생성 길이에서 요청한 수만큼 정확히 생성됐고, ON/OFF 본측정 10회 token IDs가 모두 일치했다.
평균 ± 표본 표준편차 (`n=5`, `ddof=1`)는 다음과 같다.

| 생성 토큰 | OFF latency (s) | ON latency (s) | Speedup |
|---:|---:|---:|---:|
| 32 | 0.7978 ± 0.0016 | 0.8032 ± 0.0023 | 0.993× |
| 64 | 1.6040 ± 0.0022 | 1.6093 ± 0.0081 | 0.997× |
| 128 | 3.2091 ± 0.0020 | 3.2159 ± 0.0037 | 0.998× |
| 256 | 6.5725 ± 0.0328 | 6.4251 ± 0.0164 | 1.023× |

생성 32·64·128개에서는 ON/OFF 시간이 비슷했고, 256개에서는 약 1.023배의 latency speedup이 관측됐다.
Prompt length sweep의 긴 입력에서 관측된 큰 개선과는 차이가 있다. 이는 입력 128, 생성 최대 256이라는 이번 범위에서의 관측이다.
반복 5회의 기술 통계이며 미세한 차이의 통계적 유의성을 주장하지 않는다.

![Generation sweep](../plots/generation_sweep.png)

[Raw CSV](../results/generation_sweep_raw.csv) · [Summary JSON](../results/generation_sweep_summary.json) · [PDF](../plots/generation_sweep.pdf)
