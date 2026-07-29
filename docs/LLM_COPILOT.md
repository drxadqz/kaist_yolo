# Experiment Report Copilot (optional)

The experiment copilot is a small, optional **result-to-report** utility. It reads
`results/benchmark_summary.csv`, separates evidence by claim scope, and prepares
an auditable prompt or JSON context. Its default mode is completely local: no
network request, API key, OpenAI package, or Pydantic installation is required.

This utility is not part of DARP-Net training or inference. The detector does not
use an LLM, and none of its detection metrics should be attributed to an LLM.
The only LLM-backed operation is the optional conversion of an existing result
table into a first-draft report after the user explicitly supplies
`--call-openai`.

## What the copilot enforces

Every CSV row is assigned to one of five evidence groups:

| `setting` / normalized scope | Intended meaning | Reporting rule |
| --- | --- | --- |
| `single_model` | Formal detector/checkpoint evidence | May be described only as a single-model result under its recorded protocol. |
| `system` | Routed, ensembled, or otherwise system-level evidence | Must not be presented as the formal single model. |
| `postprocess` | A result whose change comes from post-processing | Must retain the post-processing qualifier. |
| `protocol_optimized` | A protocol-aware or protocol-specific optimized setting | Must retain the protocol-specific qualifier and must not imply generic transfer. |
| `unspecified` | Missing or unrecognized scope | Cannot become a verified claim until manually classified. |

The generated context includes the CSV row number, source-file SHA-256 digest,
raw normalized fields, scope counts, and reporting guardrails. The prompt tells
the model to preserve metric names, values, units, split names, protocols, and
scope boundaries. It also treats CSV text as untrusted evidence rather than as
instructions. These controls reduce accidental overclaiming, but an LLM draft
still requires human review against the result cards and evaluation logs.

## Local-only usage (default)

Generate the reviewable prompt on standard output:

```bash
python scripts/experiment_copilot.py
```

Write a machine-readable evidence package without any network access:

```bash
python scripts/experiment_copilot.py \
  --format context-json \
  --output artifacts/experiment_context.json
```

Use a different CSV explicitly:

```bash
python scripts/experiment_copilot.py \
  --input-csv results/benchmark_summary.csv \
  --output artifacts/experiment_prompt.txt
```

`--output` is optional. Without it, the utility writes to stdout. The local path
uses only the Python standard library.

## Optional OpenAI structured-output draft

Install the optional dependencies:

```bash
python -m pip install -r requirements-llm.txt
```

Set `OPENAI_API_KEY` in the process environment using your operating system or
secret manager, then opt in explicitly:

```bash
python scripts/experiment_copilot.py \
  --call-openai \
  --output artifacts/experiment_report.json
```

The script never accepts an API key as a command-line argument and never prints
or writes the key. `OpenAI()` lets the official SDK read `OPENAI_API_KEY` from
the environment. The default model is
`OPENAI_MODEL` if set, otherwise `gpt-5.6-luna`; `--model` overrides it for one
run:

```bash
python scripts/experiment_copilot.py \
  --call-openai \
  --model gpt-5.6-luna \
  --output artifacts/experiment_report.json
```

Only this explicit mode sends the generated benchmark evidence to the API. Do
not use it for confidential metrics unless that data transfer is authorized.
The implementation uses the OpenAI Responses API and
`client.responses.parse(..., text_format=ExperimentReport)` with a Pydantic
schema. The structured JSON contains:

- `executive_summary`
- `verified_claims`
- `caveats`
- `resume_bullets`
- `interview_questions`

See the official OpenAI documentation for the
[Responses API and text generation](https://developers.openai.com/api/docs/guides/text)
and [structured model outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

## CSV contract

The file must be UTF-8 CSV with a header and at least one non-empty data row.
Column names are normalized to lowercase `snake_case`. Use `setting` as the
preferred scope column. The parser also recognizes `claim_scope`, `scope`, `result_scope`,
`evidence_scope`, `evaluation_scope`, `claim_type`, `result_type`,
`result_class`, `category`, and `track` for compatibility. Unknown values remain
`unspecified`; the utility deliberately does not guess scope from a model name
or free-text note.

The canonical table also carries `evidence_level` and `claim_boundary`. Both are
preserved verbatim in the context: an evidence level describes provenance, while
a claim boundary is a hard reporting limit that the generated draft must retain.

Keep the following fields explicit whenever they apply:

- experiment/model name and checkpoint identity;
- metric name, value, and unit (for example, MR in percent rather than fraction);
- evaluation protocol, split, subset, and seed;
- whether calibration, routing, post-processing, or protocol-specific flags are
  enabled;
- evidence status and a result-card/config/log reference.

## 中文说明

实验报告助手只是一个可选的“结果表转报告”工具，并不是检测器的一部分。DARP-Net
的训练、融合与推理过程不调用大语言模型，因此不能把任何检测指标写成“LLM
带来的提升”。默认命令只在本地读取 CSV、生成提示词或 JSON，不联网，也不需要
API Key。

只有显式添加 `--call-openai` 时，脚本才会把生成的指标上下文发送给 OpenAI
Responses API，并要求模型返回结构化的报告草稿。API Key 只能通过
`OPENAI_API_KEY` 环境变量交给官方 SDK；脚本不会从命令行接收、记录或打印密钥。
调用前请确认指标数据允许传输，并在生成后逐条对照 result card、配置和日志复核。

报告时必须严格区分 `single_model`、`system`、`postprocess` 和
`protocol_optimized`。未识别的行会进入 `unspecified`，在人工补全分类前不能用于
简历中的强结论。这一设计的目标是让 LLM 帮助整理证据，而不是用 LLM 包装或夸大
原有视觉模型的结果。
