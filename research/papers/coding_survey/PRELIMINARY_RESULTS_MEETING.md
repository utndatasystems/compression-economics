# Preliminary optimization results — meeting brief

## Headline result

![Matched 10 MiB text8 round-trip timing](manuscript/figures/text8_roundtrip_timing.png)

On the matched 10 MiB `text8` workload (Qwen2.5-0.5B, H100, batch 256,
context 256, retain 100), CUDA v2 improves complete compression throughput
from 0.0854 to 0.1002 MiB/s (1.17x). Its arithmetic-coding stage is 422x
faster, but model inference now dominates. CUDA-CDF has the larger end-to-end
effect on decompression: 102.6 s rather than 629.7 s for the host baseline
(6.14x), with a 51.8x faster arithmetic stage. Every measured decoder exactly
recovers the source; the target-interval encoders produce byte-identical
archives at a charged compression ratio of 7.682x. Values are medians of three
timed runs after one full-workload warm-up.

## Throughput–compression operating points

![Measured throughput/compression operating points](manuscript/figures/optimization_operating_points.png)

The high-parallelism batch-16,384 configuration reaches 0.5823 MiB/s encoding
and 0.4787 MiB/s decoding—5.81x and 4.91x the batch-256 rates. This is not a
free optimization: its charged compression ratio is 5.498x instead of 7.682x
because each of the 16,384 streams has less usable context and needs a seed.

## Current optimization state

| Optimization | Current evidence | Result | Remaining limit |
|---|---|---:|---|
| CUDA v1 encoding | Complete 10 MiB round trip; three runs | 1.14x full path; 404x arithmetic stage | Model inference becomes dominant. |
| CUDA v2 fused intervals | Complete 10 MiB round trip; three runs | **1.17x full path; 422x arithmetic stage**; interval construction 7.87x faster than Python | Full encoding is now mostly inference-bound. |
| PyTorch device decoding | Complete 10 MiB round trip; three runs | 4.32x full path; 9.86x arithmetic stage | Fine-grained tensor operations and launches remain expensive. |
| CUDA-CDF decoding | Complete 10 MiB round trip; three runs | **6.14x full path; 51.8x arithmetic stage**; 1.42x full path over PyTorch device | CDF construction remains outside the fused kernel. |
| Parallel operating point | Complete 10 MiB round trip; three runs | 0.5823 MiB/s encode; 0.4787 MiB/s decode | Ratio falls from 7.682x to 5.498x. |
| Frozen CUDA-CDF scaling | Five-run frozen traces | At 16 streams: 103.36 ksymbols/s, 6.64x host and 18.13x PyTorch device | One stream is still slower than host; 16,896 streams is only an occupancy stress point. |
| CUDA v3 compact intervals | One-run 2,048-token pilot | 1.05x interval construction; 1.002x compression phase over v2 | Repeat runs and measure peak memory before making a paper claim. |

All complete live target-interval encoders produce byte-identical archives,
and all reported decoders exactly recover the source. The main unfinished
items are a matched full-logits baseline, fused CDF construction plus decode,
physical PCIe and peak-memory counters, and replication across models,
datasets, and hardware.

## External positioning

| Study | Hardware and operating point | Reported rate | What the result says |
|---|---|---:|---|
| [Adams et al., ADMS'26](https://www.vldb.org/2026/Workshops/VLDB-Workshops-2026/ADMS/ADMS26_3.pdf) | A100 40 GB; `text8`; Qwen2.5-0.5B; batch 256; target-interval AC; ratio 7.1007x | 0.0349 MB/s full compression | Target-interval capture reduces AC from 97.1% to 1.0% of runtime and gives 18.96x over their raw-AC baseline. Most experiments are single runs. |
| [Kipf et al., CIDR'26](https://www.vldb.org/cidrdb/papers/2026/p34-kipf.pdf) | L40S; first 1M `text8` tokens; Qwen2.5-0.5B; batch 256; context 128; ratio 7.77x | 57.52 KiB/s inference only | KV caching raises inference from 2.58 to 57.52 KiB/s. Context 1024 reaches ratio 8.01x but more than halves throughput. Their cost model estimates 10 years to beat uncompressed storage and 120 years to beat traditional compression. |

The absolute rates above are not an apples-to-apples leaderboard: hardware,
input extent, context policy, software, repetition count, and timing scope
differ. The robust shared conclusion is the bottleneck shift. CIDR'26 shows
that KV caching first makes model inference viable; ADMS'26 shows that moving
from full probability vectors to target intervals can remove a host coding
bottleneck; the current experiments show that once encoding is mostly
inference-bound, the larger remaining systems win is compiled, batched
decompression.

## Suggested meeting takeaway

> We reproduce the prior bottleneck-shift story under a stricter matched
> round-trip protocol, then extend it to decoding: GPU coding makes encoding
> 17% faster, but GPU decoding makes the full read path 6.1x faster. The next
> optimization target is no longer arithmetic encoding; it is model inference
> and fused CDF construction on decode.

## Caveats to say out loud

- These are preliminary H100 results for one model, dataset, and batch/context
  configuration.
- The 422x and 51.8x numbers are isolated arithmetic-stage speedups; use 1.17x
  and 6.14x for end-to-end claims.
- The external rates should not be divided into ours to claim a cross-paper
  speedup.
- The model and tokenizer are shared external state and are not charged to the
  7.682x archive ratio.
