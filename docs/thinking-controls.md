# Qwen3.8 / Strata thinking compatibility

The immutable model tokenizer template is
[the locked revision's tokenizer_config.json](https://huggingface.co/turboderp/Qwen3.8-27B-exl3/resolve/dfe2f4fd71bf9a1994051d4d47e6460fa96dd723/tokenizer_config.json).
Its SHA-256 and offline fixture provenance are recorded in
[`tests/fixtures/README.md`](../tests/fixtures/README.md). When thinking is enabled,
it accepts only `low`, `medium`, `xhigh`, and defaults to `xhigh` when effort is
omitted. It represents thinking off with `enable_thinking: false`.

The maintained `tabbyapi-thinking.patch` adds a model config field,
`qwen38_thinking_compat` (default false), and normalizes the final resolved
template variables when that field is enabled. Root `config.yml` enables it
for this deployment. Set it false for a model with different effort semantics.

| Resolved input | Template variables |
| --- | --- |
| `high` | `reasoning_effort: xhigh` |
| `none`, with no thinking toggle | Effort removed; `enable_thinking: false` |
| `none`, with explicit toggle | Effort removed; explicit `enable_thinking` preserved |
| `low`, `medium`, `xhigh` | Passed through |
| Effort omitted | Template default retained |
| Other effort | Passed to the template, which rejects unsupported values |

Supported request surfaces are top-level `reasoning_effort` / `enable_thinking`,
`reasoning: {effort, enabled}`, `template_vars`, and its alias
`chat_template_kwargs`. Existing upstream precedence remains:
model defaults < reasoning object < top-level fields < template variables <
model force settings. Normalization runs after that merge. Therefore an
explicit thinking toggle from any surviving source, including model defaults
or force settings, wins over the inferred `none` toggle. If thinking is
explicitly enabled alongside `none`, effort is removed and the template uses
its default `xhigh`; this is intentional conflict handling, not a new effort.
Explicitly set `enable_thinking: false` for an unambiguous off request.

Examples using the same body when switching from Strata:

```json
{"reasoning_effort": "high"}
{"reasoning_effort": "none"}
{"chat_template_kwargs": {"reasoning_effort": "medium", "enable_thinking": true}}
```

This mapping follows the sibling Strata source's `high` → `xhigh` alias and
thinking-off `none` behavior, while preserving TabbyAPI's explicit control
precedence. It does not change shared Strata settings, sampling defaults,
token budgets, the immutable model revision, or the model template. TabbyAPI's
reasoning parser remains enabled to split reasoning and answer content;
that setting is independent of whether a request enables model thinking.

Effort is prompt steering, not a hard reasoning-token allowance. Use an
explicit reasoning budget separately if required. Offline request/template
tests establish compatibility; inference quality and latency remain untested.
