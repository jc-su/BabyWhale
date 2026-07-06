# Baby Whale v4 Training Dataset Plan

Status date: 2026-05-08

Goal: build a Mac-sized educational dataset recipe that exercises the full lifecycle: pretrain, mid-train, SFT, DPO, GRPO/RL, function calling, and agent/skill use.

This project should use small, auditable slices first. The target is not to ingest hundreds of GB on a laptop. The target is to make every training stage real and reproducible.

## Dataset Policy

- Use public datasets with clear dataset cards and licenses.
- Materialize small subsets into local JSONL before training.
- Store the exact dataset id, subset/config, split, row range or seed, source hash, and license notes beside each generated file.
- Keep test/eval contamination visible. Do not train on evaluation splits.
- Keep tool-call traces schema-valid. Invalid JSON or unknown tools should be rejected during data prep, not silently repaired.
- Mix in negative examples where the correct behavior is **not** to call a tool.

## Recommended Source Datasets

| Stage | Dataset | Use | Why |
| --- | --- | --- | --- |
| Pretrain | `HuggingFaceTB/smollm-corpus`, `fineweb-edu-dedup` | general educational text | SmolLM-Corpus is designed for small language models and includes a 220B-token FineWeb-Edu deduplicated subset. |
| Pretrain | `HuggingFaceTB/smollm-corpus`, `cosmopedia-v2` | synthetic textbook/blog/story text | Large synthetic educational corpus with prompt/text fields. |
| Code mid-train | `HuggingFaceTB/smollm-corpus`, `python-edu` | Python/code continuation | Educationally scored Python files from The Stack v2; defer until downloader is implemented because content is retrieved from Software Heritage S3. |
| SFT | `HuggingFaceTB/smol-smoltalk` | small-model chat/instruction tuning | 485k-row Apache-2.0 small-model subset of SmolTalk. |
| SFT | `HuggingFaceTB/smoltalk`, `smol-constraints` | instruction-following constraints | Trains exact-format and constraint obedience. |
| SFT | `HuggingFaceTB/smoltalk`, `self-oss-instruct` | coding assistant behavior | Adds code-generation style prompts. |
| Tool SFT | `HuggingFaceTB/smoltalk`, `apigen-80k` | function-calling examples | SmolTalk includes APIGen function-calling samples. |
| Tool SFT | `Salesforce/xlam-function-calling-60k` | schema-grounded function calls | 60k APIGen/xLAM examples with tools and answers; gated acceptance and CC-BY-4.0. |
| DPO | `allenai/llama-3.1-tulu-3-8b-preference-mixture` | chosen/rejected preference pairs | Tulu 3 collection provides full DPO preference mixtures. |
| DPO/RM | UltraFeedback-derived data | broad preference training | UltraFeedback is a widely used preference dataset with 64k prompts and 256k responses. |
| GRPO/RLVR | `allenai/RLVR-GSM`, `allenai/RLVR-MATH`, `allenai/RLVR-IFeval` | verifiable math/instruction rewards | Tulu 3 released RLVR datasets for verifiable outcomes. |
| Tool eval | BFCL | function-call exact-match eval | Berkeley Function Calling Leaderboard evaluates function-calling and agentic behavior. |
| Agent eval | tau-bench-style local tasks | multi-turn tool-user-agent eval | Use the benchmark idea, but implement a tiny local domain first. |

## First Mac-Sized Recipe

Start with this exact educational mix:

| Stage | Rows | Sequence length | Source mix |
| --- | ---: | ---: | --- |
| Pretrain smoke | 2k docs | 256 | 70% `fineweb-edu-dedup`, 30% `cosmopedia-v2` |
| Pretrain small | 50k docs | 512 | 60% `fineweb-edu-dedup`, 30% `cosmopedia-v2`, 10% local project docs/code |
| Mid-train small | 20k examples | 512-1024 | 40% code, 30% math/reasoning, 30% tool/agent prompts |
| SFT small | 20k conversations | 512-1024 | `smol-smoltalk`, plus `smol-constraints`, `self-oss-instruct`, `systemchats-30k` |
| Tool SFT | 5k-20k conversations | 1024 | `apigen-80k` and accepted `xlam-function-calling-60k` |
| DPO small | 5k pairs | 512-1024 | Tulu 3 8B preference mix or cleaned UltraFeedback-derived pairs |
| GRPO small | 1k prompts | 256-512 prompt, 64 response | GSM/math/IF plus local tool tasks |

The first real run should not exceed what can be debugged in minutes. Increase rows only after:

- loss decreases.
- checkpoint resume works.
- generated samples show the target behavior.
- eval scripts can detect regressions.

## Agent And Tool-Use Format

Use one canonical transcript format for SFT, DPO, and GRPO:

```json
{"role":"system","content":"You are Baby Whale. Use tools only when needed."}
{"role":"user","content":"What is 19 * 23? Use a tool if helpful."}
{"role":"assistant","content":"<tool_call>{\"name\":\"calculator.multiply\",\"arguments\":{\"a\":19,\"b\":23}}</tool_call>"}
{"role":"tool","name":"calculator.multiply","content":"{\"ok\":true,\"result\":437}"}
{"role":"assistant","content":"437"}
```

Rules:

- Tool calls must be strict JSON inside `<tool_call>...</tool_call>`.
- Tool result messages must never be predicted by the model during normal assistant generation.
- SFT loss should apply only to assistant tokens, not system/user/tool-result tokens.
- The same schema should be used by the local tool runtime and by training data validation.
- Include examples where the model should answer directly without a tool.

## Local Educational Tool Registry

Start with deterministic tools so GRPO rewards are verifiable:

| Tool | Purpose | Reward checks |
| --- | --- | --- |
| `calculator.add/subtract/multiply/divide` | arithmetic | exact result, valid JSON, required args |
| `calculator.solve_linear` | simple algebra | exact numeric answer |
| `string.search/replace/count` | text editing | exact string output |
| `table.lookup/filter/sum` | structured data | exact table result |
| `calendar.day_of_week` | date reasoning | exact date result |
| `python.run_tests` | tiny coding tasks | all tests pass in a sandbox |
| `project.search_refs` | educational local docs search | answer cites retrieved doc id |

Do not start with live web/API tools. Live tools make correctness hard to reproduce on a laptop. Add real API tools only after the local schema and reward path are stable.

## Training Stages

### 1. Pretrain

Objective:

- next-token prediction over educational text/code.

Data:

- small materialized slices from SmolLM-Corpus.

Gate:

- validation loss decreases.
- checkpoint resume is exact enough to continue training.
- generated text is syntactically stable.

### 2. Mid-Train

Objective:

- bias the base model toward code, math, long-context snippets, and agent/tool syntax before chat SFT.

Data:

- code examples.
- math/reasoning examples.
- tool schema descriptions and synthetic tool-call traces.

Gate:

- tool-call tokens become structurally likely.
- code/math validation loss improves versus pretrain-only checkpoint.

### 3. SFT

Objective:

- teach chat format, instruction following, system prompt following, tool-call syntax, and final-answer synthesis.

Data:

- `smol-smoltalk` for small-model instruction behavior.
- SmolTalk APIGen subset and xLAM function-calling for tools.

Gate:

- assistant-only loss masking is audited.
- JSON tool calls parse.
- no-tool examples do not over-call tools.

### 4. DPO

Objective:

- prefer concise, correct, schema-valid, non-hallucinated answers.

Data:

- Tulu 3 preference mix or UltraFeedback-derived pairs.
- project-generated tool preference pairs:
  - valid call beats invalid JSON.
  - correct args beat wrong args.
  - no tool beats unnecessary tool.
  - final answer using tool result beats ignoring tool result.

Gate:

- DPO loss matches toy hand checks.
- held-out preference accuracy improves.

### 5. GRPO / RLVR

Objective:

- improve verifiable math and tool-use behavior through rewards.

Reward components:

- `+1` valid JSON tool call.
- `+1` tool name exists.
- `+1` required args present with correct types.
- `+1` tool result correct.
- `+1` final answer correct.
- penalties for hallucinated tool names, invalid JSON, calling tools when forbidden, or ignoring tool output.

Gate:

- pass rate improves on held-out math/tool tasks.
- JSON validity and schema validity are reported separately from final answer accuracy.

## Implementation Work Needed

1. Add `baby_whale_v4.data.hf_prepare`: done for generic subset materialization; still needs source-specific adapters.
   - stream/load named datasets.
   - normalize source rows into `pretrain`, `chat`, `preference`, or `tool_trace` JSONL.
   - write a manifest with dataset id, subset, split, row count, seed, source URL, and license note.
2. Add a trained tokenizer path:
   - train from the local materialized corpus.
   - persist tokenizer config and hash.
   - reject checkpoints with mismatched tokenizer hash.
3. Add tool schema dataclasses: done for the first strict tool-call surface.
   - `ToolSpec`, `ToolCall`, `ToolResult`, `ToolTrace`.
   - strict JSON parser.
   - no unknown tool names.
4. Add local deterministic tools: started with calculator, string, and calendar tools.
   - calculator, string, table, calendar, project-doc search.
5. Add evals: started with reward breakdown metrics for tool calls.
   - function-call exact match.
   - schema validity.
   - tool-result final-answer consistency.
   - mini tau-bench-style multi-turn local tasks.

## Sources

- SmolLM-Corpus: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus
- SmolTalk: https://huggingface.co/datasets/HuggingFaceTB/smoltalk
- Smol-SmolTalk: https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
- APIGen/xLAM function calling: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
- ToolBench: https://github.com/OpenBMB/ToolBench
- Tulu 3 datasets: https://huggingface.co/collections/allenai/tulu-3-datasets
- Tulu 3 report/blog resources: https://allenai.org/blog/tulu-3-technical
- UltraFeedback: https://github.com/OpenBMB/UltraFeedback
- Berkeley Function Calling Leaderboard: https://gorilla.cs.berkeley.edu/leaderboard
- tau-bench paper: https://arxiv.org/abs/2406.12045
