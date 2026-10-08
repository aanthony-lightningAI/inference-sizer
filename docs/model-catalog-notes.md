# Model catalog notes — verification record (2026-10-08)

Provenance for every preset added in the extended catalog. Numbers come from
official config.json / model cards where marked **published**; family-inherited
or third-party-derived values are marked **unverified** with the reason.
Requested combinations that do not exist publicly are listed at the end, never
fabricated.

KV formulae used by the engine (see `inference-sizer/sizer/engine.py`):

- `mha_gqa`: `2 x layers_with_kv x kv_heads x head_dim x kv_bytes`
- `mla`: `layers_with_kv x (kv_lora_rank + qk_rope_head_dim) x kv_bytes`,
  replicated across TP (matches vLLM/CloudTP layouts, e.g. Mooncake page math)
- `override`: vendor- or config-derived per-token figure, cited per preset

Hybrid models add a per-sequence constant (`kv_fixed_bytes_per_sequence`)
covering sliding-window caps (capped at the window), linear/DeltaNet recurrent
state, and compressed sparse pools. Derivations are stated below; where the
state dtype is not published the derivation is marked unverified.

## DeepSeek

### deepseek_v4_flash_0731 — published
Source: HF `deepseek-ai/DeepSeek-V4-Flash-0731` config.json + model card;
Frontier ML spec page (values read from repo config + safetensors index).
- `DeepseekV4ForCausalLM`, MoE: 43 layers, hidden 4096, 64 Q heads / 1 KV head,
  head_dim 512, 256 routed experts (6 active) + 1 shared, moe_inter 2048,
  context 1,048,576, sliding window 128, hybrid CSA/HCA sparse attention
  (per-layer `compress_ratios`), FP4 experts, FP8 activation, DSpark MTP.
- Params: 304B total (safetensors count); model card says Flash = 285B/13B
  active. Catalog uses 304B with a note on the discrepancy; active 13B.
- KV: override 90,112 B/token (88 KiB at fp16, third-party derivation from the
  repo cache layout; the CSA/HCA formula is not modeled first-principles).
- Not MLA: `kv_lora_rank` does not exist in this architecture.

### deepseek_v4_1_flash — unverified KV, published arch
Source: HF README + arXiv 2609.19969 + NVIDIA NeMo `deepseek_v41` config.
- CED architecture: 40 layers (20 causal encoder + 20 decoder), hidden 4096,
  64 heads / 1 KV, head_dim 512, q_lora 1280, 384 routed (6 active) + 1 shared,
  moe_inter 2304, context 1,048,576, sliding window 128, CSA2 sparse attention.
- Params: 552B backbone; 8B active prefill / 16B decode. Plus Engram conditional
  memory (196B, sparsely accessed — excluded from weights, noted).
- KV: override 11,264 B/token — DERIVED from the vendor statement that
  persistent KV is "roughly 1/8 of DeepSeek-V4-Flash" (SWA Bounded Replay).
  Not an official per-token figure; treated as unverified.

## GLM

### glm_5, glm_5_1, glm_5_2 — 5.2 published; 5/5.1 unverified (family-inherited)
Source: HF `zai-org/GLM-5.2` config.json; `zai-org/GLM-5` GitHub release table.
- `GlmMoeDsaForCausalLM`, MoE + MLA-style compressed KV + DeepSeek Sparse
  Attention: 78 layers, hidden 6144, kv_lora_rank 512, qk_rope_head_dim 64,
  qk_nope 192, q_lora 2048, 64 heads, 256 routed experts (8 active) + 1 shared,
  moe_inter 2048, context 1,048,576, index_topk 2048.
- Params: 744B total / 40B active (all of GLM-5, 5.1, 5.2, 5.3 share this size
  class per the official table).
- KV: MLA path — 78 x (512+64) x 2 = 89,856 B/token. Mooncake's `glm5` layout
  confirms the same per-token math.
- GLM-5 and GLM-5.1 configs were not individually verified; they inherit the
  verified GLM-5.2 numbers and are marked unverified for that reason.

### glm_5_3_flash — published
Source: HF `zai-org/GLM-5.3-Flash` config.json; z.ai blog.
- `Glm5NextForConditionalGeneration` (multimodal), MoE hybrid: 45 layers —
  34 linear-attention (KDA) + 11 full DSA layers (`full_attn_layers`
  3,7,...,43), hidden 4096, kv_lora_rank 512, qk_nope_head_dim 256,
  qk_rope_head_dim 0, 64 heads, 288 routed experts (8 active) + 1 shared,
  moe_inter 2048, context 1,048,576, IndexPool compression.
- Params: 320B total / 18B active.
- KV: MLA path on the 11 full layers — 11 x (512+0) x 2 = 11,264 B/token;
  fixed KDA state term derived as 34 layers x 64 heads x 128 x 128 x 2B =
  71,303,168 B/sequence (dtype unverified — bf16 assumed).

## Kimi (Moonshot)

### kimi_k2_6, kimi_k2_7_code — published
Source: HF model cards (`moonshotai/Kimi-K2.6`, `Kimi-K2.7-Code`);
Mooncake MLA layout table; maxText kimi-k2 config.
- MoE + MLA: 1T total / 32B active, 61 layers (1 dense), 384 routed experts
  (8 active) + 1 shared, moe_inter 2048, hidden 7168, 64 heads, MLA
  kv_lora_rank 512 + qk_rope_head_dim 64, context 262,144.
- KV: MLA path — 61 x (512+64) x 2 = 70,272 B/token (Mooncake: 1,152 B/token).
- K2.7-Code reuses the K2.6 architecture (official statement).

### kimi_k3 — published
Source: HF `moonshotai/Kimi-K3` config.json + README; Moonshot GitHub.
- MoE hybrid: 2.8T total / 104B active, 93 layers — 69 KDA linear + 24 gated
  MLA layers (every 4th + final), 96 heads, hidden 7168, MLA kv_lora_rank 512 +
  qk_rope_head_dim 64 (NoPE elsewhere), 896 routed experts (16 active) + 2
  shared, moe_inter 3072, context 1,048,576, MXFP4 expert weights (QAT).
- KV: MLA path on 24 layers — 24 x (512+64) x 2 = 27,648 B/token; fixed KDA
  state derived as 69 layers x 96 heads x 128 x 128 x 4B = 434,110,464 B/seq
  (fp32 assumed per `mamba_ssm_dtype`; dtype unverified).

## Qwen

Verified releases (QwenLM/Qwen3.8 repo + HF). All use hybrid Gated DeltaNet
(linear attention) + periodic full/sparse attention; only full-attention
layers carry a growing KV cache; linear layers hold constant recurrent state
(included as fixed per-sequence term, fp32 assumed per `mamba_ssm_dtype`).

### qwen_3_5_27b — published
32 layers: 8 full-attention (16Q/4KV, head_dim 256) + 24 linear (16K/32V
heads, 128); dense 27B, hidden 4096, context 262,144.
KV: 2 x 8 x 4 x 256 x 2 = 32,768 B/token.
Source: HF transformers `Qwen3_5TextConfig` defaults + model card.

### qwen_3_5_35b_a3b — published
40 layers: 10 full (16Q/2KV, 256) + 30 linear; MoE 256 experts (8 routed + 1
shared), moe_inter 512, hidden 2048; 35B total / 3B active (advertised;
computed split ~34.6B/2.86B), context 262,144.
KV: 2 x 10 x 2 x 256 x 2 = 20,480 B/token.
Source: HF `Qwen3_5MoeTextConfig` + inference-lab config derivation.

### qwen_3_5_122b_a10b — published
48 layers: 12 full (32Q/2KV, 256) + 36 linear (64V/16QK, 128); MoE 256 experts
(8+1), moe_inter 1024, hidden 3072; 122B/10B, context 262,144.
KV: 2 x 12 x 2 x 256 x 2 = 24,576 B/token. Linear state: 36 x 64 x 128 x 128 x
4B = 150,994,944 B/seq (dtype unverified).
Source: HF model card (layout table).

### qwen_3_6_27b — published
64 layers: 16 full (24Q/4KV, 256) + 48 linear (48V/16QK, 128); dense 27B,
hidden 5120, context 262,144. KV: 2 x 16 x 4 x 256 x 2 = 65,536 B/token.
Source: HF config.json + README.

### qwen_3_8_27b — published
Same layout as Qwen3.6-27B (64 layers, 16 full 24Q/4KV + 48 linear); dense
27B, hidden 5120, context 262,144 native (1M via YaRN). KV: 65,536 B/token.
Source: HF README (layout table) + model card.

### qwen_3_8_flash_next — published
48 layers: 12 Qwen Sparse Attention (24Q/2KV, head_dim 256,
`indexer_compress_ratio` 4) + 36 DeltaNet; MoE 125B total / 6B active (plus
51B n-gram embeddings and 4B MTP, excluded from weights with note), hidden
2560, moe_inter 640, context 262,144.
KV: sparse layers store pooled entries at compress ratio 4 —
12 x 2 x 2 x 256 x 2 / 4 = 6,144 B/token; DeltaNet state 36 x 48 x 128 x 128 x
4B = 113,246,208 B/seq (dtype unverified).
Source: HF config.json + README (layout table).

## Inkling (Thinking Machines)

### inkling — published
Source: HF `thinkingmachines/Inkling` config.json + transformers
`configuration_inkling.py` + HF blog.
- `InklingForConditionalGeneration` (multimodal), MoE: 975B total / 41B
  active, 66 layers — 55 sliding-window + 11 global (5:1, final global),
  hidden 6144; global 64Q/8KV head_dim 128; sliding 64Q/16KV head_dim 128,
  window 512; 256 routed experts (6 active) + 2 shared, moe_inter 3072,
  context 1,048,576 (model_max_length).
- KV: growing (global) 2 x 11 x 8 x 128 x 2 = 45,056 B/token; sliding fixed
  2 x 55 x 16 x 128 x 2 x 512 = 230,686,720 B/seq.

### inkling_small — published
Same architecture; 276B total / 12B active (HF blog release note).

## Muse Spark 1.3 — NOT ADDED (no public specification)
Meta proprietary multimodal model; API-only, no published weights, parameter
count, layers, or attention configuration (Artificial Analysis / third-party
pages confirm "parameters: not available"). A sizing preset would require
fabricated architecture numbers, so it is excluded. If Meta publishes specs or
weights, add via the config.json import path.

## Gemma (current generation = Gemma 4)

### gemma_4_31b — published
Source: HF `google/gemma-4-31B` config.json + Google model card + Raschka notes.
- Dense multimodal: 30.7B params, 60 layers (50 sliding + 10 global, 5:1,
  final global), hidden 5376; local 32Q/16KV head_dim 256 (window 1024);
  global 32Q/4KV head_dim 512 with unified K/V (`attention_k_eq_v`) and
  p-RoPE; context 262,144.
- KV: growing (global, unified K/V = one stream) 1 x 10 x 4 x 512 x 2 =
  40,960 B/token; sliding fixed 2 x 50 x 16 x 256 x 2 x 1024 = 838,860,800
  B/seq.

### gemma_4_26b_a4b — published
Source: HF `google/gemma-4-26B-A4B` config.json + model card.
- MoE: 25.2B total / 3.8B active, 30 layers (25 sliding + 5 global), hidden
  2816; local 16Q/8KV head_dim 256 (window 1024); global 16Q/2KV head_dim 512
  unified K/V; 128 routed experts (8 active) + 1 shared, moe_inter 704,
  context 262,144.
- KV: growing 1 x 5 x 2 x 512 x 2 = 10,240 B/token; sliding fixed
  2 x 25 x 8 x 256 x 2 x 1024 = 209,715,200 B/seq.

### E2B / E4B / 12B — not added
Edge-tier sizes (2.3B-12B) verified to exist but below the GPU fleet-sizing
threshold this tool targets; omitted deliberately (documented here, not an
oversight). Gemma 3 (previous generation) was also not requested for the
catalog.

## Requested combinations that do not exist (verified against QwenLM repos + HF)

- Qwen3.5-235B — no such release (3.5 tops out at 397B-A17B; 235B exists only
  as Qwen3-235B-A22B, generation 3.0)
- Qwen3.6-122B, Qwen3.6-235B — no such releases (3.6 shipped 27B and 35B-A3B)
- Qwen3.8-35B, Qwen3.8-122B, Qwen3.8-235B — confirmed absent from the Qwen HF
  organization (3.8 shipped 27B, Flash-Next, 2.4T-A95B, plus API-only SKUs)
- Muse Spark 1.3 — exists but proprietary (see above)
- "GLM 5.3 Flash" interpretation: the Flash variant exists at 5.3 only;
  GLM-5 / 5.1 / 5.2 are the 744B-A40B flagship models (GLM-5.3 744B also
  exists but was not requested).