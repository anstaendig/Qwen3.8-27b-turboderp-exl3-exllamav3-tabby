# Setup model catalog evidence

## Result

`setup/models.json` lists all 25 model-bearing EXL3 quant branches currently returned for `turboderp/Qwen3.8-27B-exl3`: 18 self-calibrated (`SC_`) variants and 7 plain bpw variants. It includes the H6 and H6 V6 options at 5.00 and 6.00 bpw. The `main` branch is excluded because its revision metadata listed `cal_trace.safetensors` but no model weight safetensors. This catalog is a point-in-time selection list; the immutable 40-character commit IDs make its entries reproducible.

## Method

On 2026-09-26, metadata-only HTTPS requests were made to:

- `https://huggingface.co/api/models/turboderp/Qwen3.8-27B-exl3/refs`
- `https://huggingface.co/api/models/turboderp/Qwen3.8-27B-exl3/revision/{commit}` for each branch's `targetCommit`

The refs response contained 26 branches: `main` plus 25 quant branches. Each revision response was checked for the `exl3` tag and at least one `.safetensors` model file other than the calibration trace. The response listed model shard files (including monolithic `model.safetensors` cases) and associated metadata; no file content or model weight was requested. The revision SHA in each catalog entry is exactly the branch's `targetCommit` from the refs response. The model `directory` is derived from its branch and restricted to lowercase alphanumeric characters and hyphens.

To repeat the check without fetching weights:

```sh
python3 - <<'PY'
import json, urllib.request
base = 'https://huggingface.co/api/models/turboderp/Qwen3.8-27B-exl3'
refs = json.load(urllib.request.urlopen(base + '/refs'))
for branch in refs['branches']:
    sha = branch['targetCommit']
    meta = json.load(urllib.request.urlopen(base + '/revision/' + sha))
    files = [item['rfilename'] for item in meta.get('siblings', [])]
    weights = [name for name in files if name.endswith('.safetensors') and name != 'cal_trace.safetensors']
    print(branch['name'], sha, 'exl3' in meta.get('tags', []), weights)
PY
```

## Facts and unknowns

Facts from the API metadata: the repository and branch names, immutable commit IDs, `exl3` tags, and model weight filenames are present in the queried revision responses. These establish that the selected revisions expose EXL3-named model files in Hub metadata.

Unknown from this metadata-only check: whether every weight object is currently downloadable, its content integrity, shard hashes, actual quantization internals, compatibility with a particular TabbyAPI/ExLlamaV3 version, memory requirements, load success, and inference behavior. No weight downloads, host operations, or inference requests were made. The branch listing can change; rerun the refs and per-revision checks when refreshing this catalog.

## Vision and MTP metadata check

A second metadata-only pass queried the revision file listing for each of the 25 catalog entries and fetched its small `config.json` (using the immutable revision URL). All 25 list `config.json`, `tokenizer_config.json`, `preprocessor_config.json`, and `video_preprocessor_config.json`. Every `config.json` declares `architectures: ["Qwen3_5ForConditionalGeneration"]`, a `vision_config`, and image/video plus vision boundary token IDs. Thus the metadata advertises a vision-capable model configuration for every listed branch, including every plain bpw branch; no plain or SC branch was found without vision configuration. This is configuration evidence, not proof that image/video input loads or runs in the target TabbyAPI/ExLlamaV3 build.

No revision's file listing has a filename containing `mtp` or `draft`, and none of the 25 `config.json` files has a top-level config key containing `mtp` or `draft`. That does **not** establish that MTP is absent or unusable: the metadata does not enumerate tensor names inside model shards, and this check did not inspect shard contents. Therefore MTP capability remains unverified for all 25 branches from this evidence. In particular, the H6 / H6 V6 suffixes and the branch's `V` marker should not be treated as evidence of MTP support. No option can be labeled definitively MTP-unusable based only on these metadata checks.

The check fetched only `config.json` for each pinned commit, plus the repository metadata endpoint for sibling filenames; it did not fetch tokenizer contents, tensor data, weight objects, or run inference. A live or static compatibility check against the pinned ExLlamaV3 loader is still needed to determine whether an MTP head is present and recognized for any branch.
