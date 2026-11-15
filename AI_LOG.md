Module: libs/bulk_prompt.py — What it wrote: CSV prompt loading, agent-backed bulk execution, token cost accounting, and paired result/report CSV generation.
Test: Not run per request.
Reviewed: CSV schemas for prompt/language and English/Bangla pairs; model/options and output paths.

Module: libs/ollama_engine.py — What it wrote: Optional model/settings/cache-lifetime controls and aggregated prompt/output token and latency usage.
Test: Not run per request.
Reviewed: Existing interactive defaults and unchanged tool-call path.

Module: libs/printer.py — What it wrote: Prompt and completion token counts while retaining the existing total-token return.
Test: Not run per request.
Reviewed: Existing streaming output remains enabled by default; bulk can suppress it.

Module: main.py — What it wrote: `--dataset` / `--model` bulk CLI and result/report output options.
Test: Not run per request.
Reviewed: Existing interactive mode remains the default when no dataset is supplied.

Module: dataset.csv — What it wrote: 50 aligned English–Bangla pairs in CSV format for the agent bulk runner.
Test: Not run per request.
Reviewed: Kept the id, category, english, and bangla dataset columns.
