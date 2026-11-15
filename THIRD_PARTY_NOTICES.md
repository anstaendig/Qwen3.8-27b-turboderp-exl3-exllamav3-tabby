# Third-party notices and licensing

## This project

Everything in this repository **except** the `patches/` directory, the
`tabbyAPI/` submodule, and the upstream template fixture under `tests/fixtures/` is released under the [MIT License](LICENSE): the setup
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
[`47281587fafbd69e50aaf77c7626f390811126b6`](https://github.com/theroyallab/tabbyAPI/tree/47281587fafbd69e50aaf77c7626f390811126b6).
It is not vendored here; `git submodule update` fetches it from upstream.
TabbyAPI is licensed under AGPL-3.0; see
[its license](https://github.com/theroyallab/tabbyAPI/blob/47281587fafbd69e50aaf77c7626f390811126b6/LICENSE).

## Prometheus Python client

The metrics patch pins `prometheus-client==0.26.0`, licensed under
[Apache-2.0](https://github.com/prometheus/client_python/blob/v0.26.0/LICENSE).

## ExLlamaV3

TabbyAPI's `cu13` dependency selects the official ExLlamaV3 v1.6.7
`cu132.torch2.11.0` wheel alongside PyTorch 2.11.0 `cu130`. Its upstream source is
[`8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0`](https://github.com/turboderp-org/exllamav3/tree/8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0).
ExLlamaV3 is MIT-licensed and is not patched here; see the
[upstream license](https://github.com/turboderp-org/exllamav3/blob/8c593e4e5c88da76a4aaf8f1b1838bb7c1fce4e0/LICENSE).

## Model weights

Model weights are not included. The setup wizard downloads them from
[`turboderp/Qwen3.8-27B-exl3`](https://huggingface.co/turboderp/Qwen3.8-27B-exl3)
on Hugging Face; they are subject to the license stated on that model card and
on the upstream Qwen model card.

## Test template

`tests/fixtures/qwen38-chat-template.jinja` is unmodified model material from the
locked tokenizer configuration. It retains the upstream Qwen Apache-2.0 license;
see [fixture provenance](tests/fixtures/README.md).
