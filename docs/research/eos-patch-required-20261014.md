# Is the EOS patch required for the pinned TabbyAPI/ExLlamaV3?

Date: 2026-10-14

## Verdict

**For pinned TabbyAPI `a3fed43efae3b18ec6181618d6a4a66153a9d14e` with the ExLlamaV3 1.6.1 source at `f45b84512ba65bca77f7d394399d4b68821fc71b`, the patch is required if callers expect `ban_eos_token: true` / `ignore_eos: true` to ignore the backend's implicit EOS stop IDs.** In unpatched Tabby, the field parses but is classified as unsupported and ignored by the ExLlamaV3 sampler path; separately, that path always adds model and backend inferred EOS IDs to each job's stop conditions. The engine then ends the job when a sampled token matches one of those IDs.

This is a **stop/termination behavior**, not a literal token ban. The patch removes only the implicit EOS stop IDs when the flag is true. It leaves caller-supplied stop strings/token IDs active, and does not mask EOS out of the sampler. With the patch, EOS can therefore be emitted and generation can continue until another stop condition, a filter/loop stop, or a generation limit. A caller that wants a literal sampler ban can use `banned_tokens` with the relevant EOS token ID(s); that is a distinct sampler operation.

## Pin-specific source evidence

### TabbyAPI request parsing and unpatched behavior

- **Fact — accepted names/default:** `BaseSamplerRequest.stop` accepts `stop` and `stop_sequence`, and `banned_tokens` accepts `banned_tokens` and `custom_token_bans`. A validator wraps a single stop string into a list. `ban_eos_token` defaults to false and accepts the alias `ignore_eos`. See [`common/sampling.py` at the pinned TabbyAPI commit, lines 96–108, 285–290, and 474–481](https://github.com/theroyallab/tabbyAPI/blob/a3fed43efae3b18ec6181618d6a4a66153a9d14e/common/sampling.py#L96-L108).
- **Fact — no sampler implementation in the unpatched path:** pinned `UNSUPPORTED_PARAMS` contains `"ban_eos_token": False` and says activated parameters are warned about and ignored. `warn_unsupported_params` includes a non-neutral true value in that warning. See [`common/sampling.py`, lines 24–37 and 544–565](https://github.com/theroyallab/tabbyAPI/blob/a3fed43efae3b18ec6181618d6a4a66153a9d14e/common/sampling.py#L24-L37).
- **Fact — separate special-token helper does not wire the request flag:** `ExllamaV3Container.get_special_tokens` can blank the EOS spelling when its own `ban_eos_token` argument is true, but the only pinned caller invokes it with no arguments. The request field is not forwarded there. See [`backends/exllamav3/model.py`, lines 1337–1352](https://github.com/theroyallab/tabbyAPI/blob/a3fed43efae3b18ec6181618d6a4a66153a9d14e/backends/exllamav3/model.py#L1337-L1352) and [`endpoints/OAI/utils/chat_completion.py`, line 371](https://github.com/theroyallab/tabbyAPI/blob/a3fed43efae3b18ec6181618d6a4a66153a9d14e/endpoints/OAI/utils/chat_completion.py#L371).
- **Fact — actual sampler inputs:** `ExllamaV3SamplerBuilder.from_params` builds from the request parameters; its token-masking step is `params.banned_tokens` → `builder.ban_tokens(...)`. There is no `ban_eos_token` branch in the builder. See [`backends/exllamav3/model.py`, lines 1816–1830](https://github.com/theroyallab/tabbyAPI/blob/a3fed43efae3b18ec6181618d6a4a66153a9d14e/backends/exllamav3/model.py#L1816-L1830) and [`backends/exllamav3/sampler.py`, lines 113–116](https://github.com/theroyallab/tabbyAPI/blob/a3fed43efae3b18ec6181618d6a4a66153a9d14e/backends/exllamav3/sampler.py#L113-L116).
- **Fact — implicit EOS sources:** `generate_gen` starts `stop_conditions` as `params.stop`, obtains the model's EOS IDs (falling back to the tokenizer EOS ID if that list is empty), appends them, then appends `self.config.eos_token_id_list` and deduplicates. It passes the result to `AsyncJob`. See [`backends/exllamav3/model.py`, lines 1841–1856 and 2005–2014](https://github.com/theroyallab/tabbyAPI/blob/a3fed43efae3b18ec6181618d6a4a66153a9d14e/backends/exllamav3/model.py#L1841-L1856). `HFModel.eos_tokens()` combines EOS IDs from the model config and generation config; see [`common/transformers_utils.py`, lines 160–171](https://github.com/theroyallab/tabbyAPI/blob/a3fed43efae3b18ec6181618d6a4a66153a9d14e/common/transformers_utils.py#L160-L171).

### Engine job behavior

- **Fact — explicit stop conditions, no EOS default:** ExLlamaV3's `Job` accepts `stop_conditions=None`; supplied integer IDs are added to `self.stop_tokens` and strings to `self.stop_strings`. The source documents these conditions as ending generation, with the matched condition omitted from output. See [`job.py` at the pinned ExLlamaV3 source, lines 48–56 and 96–99](https://github.com/turboderp-org/exllamav3/blob/f45b84512ba65bca77f7d394399d4b68821fc71b/exllamav3/generator/job.py#L48-L56), [`lines 232–253`](https://github.com/turboderp-org/exllamav3/blob/f45b84512ba65bca77f7d394399d4b68821fc71b/exllamav3/generator/job.py#L232-L253). `Generator.generate` forwards its `stop_conditions` argument directly to `Job`; see [`generator.py`, lines 1663–1668 and 1799–1809](https://github.com/turboderp-org/exllamav3/blob/f45b84512ba65bca77f7d394399d4b68821fc71b/exllamav3/generator/generator.py#L1663-L1668). **Inference:** ExLlamaV3 does not add a model EOS default in this API path; the implicit EOS IDs here come from Tabby supplying them.
- **Fact — stop tokens terminate after sampling:** `Job` checks `token in self.stop_tokens` after accepting a sampled token and emits EOS for that job. It only masks stop tokens from logits while `min_new_tokens` has not yet been met. See [`job.py`, lines 449–465 and 968–977](https://github.com/turboderp-org/exllamav3/blob/f45b84512ba65bca77f7d394399d4b68821fc71b/exllamav3/generator/job.py#L449-L465). Thus, outside the minimum-token window, a stop token is generally sampled and then terminates; this is not the same operation as `SS_BanTokens`.
- **Fact — independent end paths remain:** the job also ends at `max_new_tokens` and can end on a completed filter, stop string, or detected loop. See [`job.py`, lines 974–981 and 1140–1167](https://github.com/turboderp-org/exllamav3/blob/f45b84512ba65bca77f7d394399d4b68821fc71b/exllamav3/generator/job.py#L974-L981).

## What the maintained patch changes

The patch adds `build_stop_conditions`. It starts from a copy of `params.stop`; when `ban_eos_token` is false it appends model and backend EOS IDs, and when true it omits those implicit IDs. It then deduplicates the combined list. The call site passes `params.ban_eos_token`; the patch also removes `ban_eos_token` from the unsupported-parameter registry. See [`patches/tabbyapi-ban-eos.patch`, lines 13–53 and 54–64](../../patches/tabbyapi-ban-eos.patch).

Consequences:

- **No flag / false:** behavior remains the pinned default: model and backend EOS IDs stop the job, along with explicit stops.
- **True / `ignore_eos: true`:** inferred model and backend EOS IDs no longer stop the job. Explicit `stop` / `stop_sequence` values remain. If an explicit stop repeats one of the EOS IDs, it still stops on that ID.
- **Literal ban:** the patch does not add EOS IDs to `banned_tokens`, alter sampler logits, or guarantee EOS cannot be produced. Use the actual EOS IDs in `banned_tokens` to forbid sampling them. The set can include multiple model/generation-config EOS IDs and backend tokenizer-derived EOS IDs.
- **Other termination:** max-token limits and non-EOS explicit stop conditions continue to apply, as do engine filter and loop stops.
- **Aliasing:** the API accepts `ignore_eos` as a validation alias for `ban_eos_token`; both select the same stop-condition behavior after the patch. The name can mislead: in this backend it means “do not implicitly terminate on EOS,” not “mask EOS from sampling.”

The patch also copies `params.stop` before extending it, unlike the pinned source's in-place `+=` operations. That avoids modifying the parsed stop list while composing per-job conditions.

## Coordinator CPU checks and upstream-head check

The coordinator executed the actual stop-assembly AST statements from unpatched
pinned TabbyAPI and the patched working source, then the pinned engine's stop-set
construction and sampled-token stop branch. Model metadata and token values were
CPU stubs; neither Torch nor the engine was imported. With synthetic model EOS
`151645` and backend EOS `[151645, 151643]`, all six cases passed:

| Flag | Explicit stops | Unpatched EOS stops on `151645` | Patched EOS stops on `151645` |
| --- | --- | --- | --- |
| false | `["END"]` | Yes | Yes |
| true | `["END"]` | Yes | No |
| true | `["END", 151645]` | Yes | Yes |

Unpatched stop assembly also mutated the request stop list in all three cases;
patched assembly left it unchanged. This is source-fragment behavior proof, not
an end-to-end API/engine test. The existing `tests/test_ban_eos.py` was run with
bytecode/cache writes disabled and returned **2 passed**.

A read-only `git ls-remote origin refs/heads/main` returned
`a3fed43efae3b18ec6181618d6a4a66153a9d14e` on 2026-10-14. Thus observed upstream
TabbyAPI main was the same revision examined here; no later TabbyAPI fix had
superseded this patch at that check.

## Scope and caveats

Evidence is static source evidence: TabbyAPI was read from the exact pinned Git commit with `git show`, rather than from its locally modified submodule worktree. ExLlamaV3 behavior was checked in the retained 1.6.1 `job.py` / `generator.py` files and against the immutable GitHub source at the stated commit. The independent reviewer performed only source inspection; the coordinator performed the CPU checks and upstream-head check recorded above. No builds, installations, model loads, inference, or host inspection were performed. This report does not claim runtime acceptance or behavior of other backend versions. Deployment files and the patch were not changed.
