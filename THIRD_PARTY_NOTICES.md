# Third-party notices and licensing

## This project

Everything in this repository **except** the `patches/` directory and the
`tabbyAPI/` submodule is released under the [MIT License](LICENSE): the setup
wizard, scripts, tests, configuration, systemd unit, and documentation.

## Patches to TabbyAPI (`patches/`)

The files in `patches/` modify TabbyAPI, which is licensed under the GNU Affero
General Public License v3.0. Because they are modifications of AGPL-3.0 code and
are applied to it, the patches are distributed under
[AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html), the same license as
TabbyAPI. If you run a patched TabbyAPI as a network service for others, the
AGPL's source-offer obligations apply to that modified server.

## TabbyAPI

The `tabbyAPI/` submodule pins upstream TabbyAPI at
[`f07131cd8fe34e449fe87cdd3a066b52b96d3cac`](https://github.com/theroyallab/tabbyAPI/tree/f07131cd8fe34e449fe87cdd3a066b52b96d3cac).
It is not vendored here; `git submodule update` fetches it from upstream.
TabbyAPI is licensed under AGPL-3.0; see
[its license](https://github.com/theroyallab/tabbyAPI/blob/f07131cd8fe34e449fe87cdd3a066b52b96d3cac/LICENSE).

## Prometheus Python client

The metrics patch pins `prometheus-client==0.26.0`, licensed under
[Apache-2.0](https://github.com/prometheus/client_python/blob/v0.26.0/LICENSE).

## ExLlamaV3

TabbyAPI's `cu13` dependency selects the official ExLlamaV3 v1.5.1
`cu132.torch2.11.0` wheel alongside PyTorch 2.11.0 `cu130`. Its upstream source is
[`958ec933361b24eb8426ec7222e5b0062a679dcd`](https://github.com/turboderp-org/exllamav3/tree/958ec933361b24eb8426ec7222e5b0062a679dcd).
ExLlamaV3 is MIT-licensed and is not patched here; see the
[upstream license](https://github.com/turboderp-org/exllamav3/blob/958ec933361b24eb8426ec7222e5b0062a679dcd/LICENSE).

## Model weights

Model weights are not included. The setup wizard downloads them from
[`turboderp/Qwen3.8-27B-exl3`](https://huggingface.co/turboderp/Qwen3.8-27B-exl3)
on Hugging Face; they are subject to the license stated on that model card and
on the upstream Qwen model card.
