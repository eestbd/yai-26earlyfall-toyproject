# Prompt length sweep

입력 128개에서 KV Cache ON/OFF 시간이 비슷했던 파일럿 결과를 확인하기 위한 길이 확장 실험이다.
기존 파일럿의 코드·입력·설정·결과는 그대로 보존한다.

## 조건

- 모델: Qwen/Qwen2.5-1.5B, revision `8faed761d45a263340a0528343f099c05c9a4323`.
- RTX 3090, BF16, eager, batch 1, greedy, seed 42, matmul TF32 off.
- 입력 길이: 128, 256, 512, 1024 (오름차순 실행); 생성 길이: 항상 128.
- 같은 모델을 한 번 로딩하고 각 길이에서 OFF 2회 → ON 2회 warm-up.
- 본측정은 길이마다 OFF/ON 각 5회이며 repeat마다 OFF→ON / ON→OFF 순서를 교대한다.
- 총 warm-up 16회와 본측정 40회. 이전 파일럿 표본은 통계에 합치지 않는다.

## 입력과 측정 범위

`data/prompt_sweep_inputs.json`은 기존 `shared_5_1_input.json`의 token IDs 128개를 각각 1·2·4·8회 반복한다.
각 짧은 입력은 다음 긴 입력의 prefix이며 tokenizer를 다시 호출하지 않는다.
길이별 token IDs와 SHA-256을 저장했고, 설정 파일 `docs/prompt_sweep_config.json`에는
원본 설정 및 input manifest 파일 해시도 고정한다.
이는 반복 텍스트를 사용한 성능 실험이다. 자연어 품질 평가나 이전 TinyLlama 실험의 정확한 재현은 아니다.

측정 경계는 기존 파일럿과 같다. GPU 입력 준비·GC·동기화 및 baseline 기록을 마친 뒤 peak를 초기화한다.
`perf_counter → generate → CUDA synchronize → perf_counter`로 전체 latency를 측정하고,
출력 검증 전에 allocated/reserved peak를 읽는다. Throughput은 `128 / total_latency_seconds`이다.
Prefill/decode 분리, decode 전용 throughput과 최종 KV tensor 크기는 이번 sweep에서 측정하지 않는다.

Warm-up에서는 매 단계의 raw logits가 유한한지, OFF가 전체 prefix를 다시 처리하고
ON이 첫 forward 이후 1개 token만 처리하는지, 실제 cache 반환 여부가 조건과 일치하는지 검증한다.
해당 hook은 본측정 전에 제거한다. 매 실행에서 생성 토큰 수와 prompt 보존을 검사한다.

모델·GPU 입력은 baseline에 포함된다. CUDA allocator cache는 실행 간·길이 간 비우지 않는다.
따라서 reserved는 앞선 실행의 영향을 받으며 GPU 전체 메모리 사용량을 뜻하지 않는다.
Peak allocated의 baseline 대비 증가량은 임시 tensor도 포함하므로 KV Cache 자체 크기가 아니다.

## 재현 및 결과 파일

```bash
python -B src/run_prompt_sweep.py
python -B src/summarize_prompt_sweep.py
python -B src/plot_prompt_sweep.py
```

- 원시 기록: `results/prompt_sweep_raw.json` (warm-up과 본측정 분리).
- 본측정 CSV: `results/prompt_sweep_raw.csv`.
- 길이별 검증·통계: `results/prompt_sweep_summary.json`.
- 그래프: `plots/prompt_sweep.png`, `plots/prompt_sweep.pdf`.
- 그래프 출처 해시: `results/prompt_sweep_plot_metadata.json`.

표준편차는 표본 표준편차 (`ddof=1`), speedup은 `OFF 평균 latency / ON 평균 latency`다.
이상치를 제거하지 않으며, 길이별 10회 token ID 배열을 첫 OFF 결과와 직접 비교한다.
출력 불일치가 있으면 최초 차이를 기록하고 비교 해석에 주의하도록 표시한다.
순서를 오름차순으로 고정한 단일 sweep이므로 길이에 따른 차이에 시간 경과·클럭 변화 등이 섞일 가능성은 남는다.
각 명령은 sweep 전용 결과 파일을 갱신한다.

## 실행 결과

길이마다 새 토큰 128개가 생성됐고, 해당 길이의 ON/OFF 본측정 10회 token IDs가 모두 일치했다.
수치는 평균 ± 표본 표준편차 (`n=5`, `ddof=1`)다.

| Prompt | OFF latency (s) | ON latency (s) | Speedup |
|---:|---:|---:|---:|
| 128 | 3.2400 ± 0.0070 | 3.2519 ± 0.0019 | 0.996× |
| 256 | 3.3763 ± 0.0040 | 3.2481 ± 0.0046 | 1.039× |
| 512 | 5.8502 ± 0.0032 | 3.2635 ± 0.0056 | 1.793× |
| 1024 | 10.9268 ± 0.0037 | 3.3005 ± 0.0051 | 3.311× |

128개 입력에서는 차이가 작지만, 512개에서 약 1.79배, 1024개에서 약 3.31배의 latency speedup이 관측됐다.
따라서 128/128 파일럿의 비슷한 시간만으로 캐시가 동작하지 않거나 효과가 없다고 결론 낼 수 없다.
측정한 반복 입력과 길이 범위에서의 관측이며 더 긴 입력·다른 모델로 일반화하지 않는다.

![Prompt sweep](../plots/prompt_sweep.png)

[Raw CSV](../results/prompt_sweep_raw.csv) · [Summary JSON](../results/prompt_sweep_summary.json) · [PDF](../plots/prompt_sweep.pdf)
