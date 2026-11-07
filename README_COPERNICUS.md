# Copernicus-57B on REAX Engine

This fork adds a Copernicus-57B runtime to the `reax13/` directory, based on REAX Engine commit `ff8f595f149abdd50ea914455eadc74cf97a277f`. The original REAX code remains available in this checkout; upstream copyright, licences and notices remain in place.

The model is distributed as `Poindexter-Labs/Copernicus-57B` with two complete directories: `part-g/` (Copernicus-G) and `part-q/` (Copernicus-Q). The runtime invokes both model-specific readouts on separate CUDA streams and uses the original `reax13/probabilities.py` mixer to average their probabilities equally. It supports text-only `choice`, `noul` and `score` requests. Request option identities are preserved and checked.

External requests are processed one at a time; requests above 16000 proxy tokens run the two models sequentially to bound activation memory. Gemma scores questions individually; Qwen groups questions with identical token lengths into batches of at most eight and 4096 tokens. Qwen uses a noncausal full-attention readout, so REAX d3 prefix caching cannot be assumed to preserve its outputs. Rendered prompts over 32,768 tokens per model are refused as unsupported; they are never truncated.

## Runtime

The default Copernicus runtime loads both models on **one NVIDIA GPU** with dynamic block-128 FP8 quantization, using the REAX FP8 kernel. Both native decision readouts are preserved. The concurrent single-GPU path completed the full Decision Index 0.3 public suite on GH200 with index 75.34. A paired 300-request sample on one RTX PRO 6000 measured 270 ms median, 827 ms mean and 839 ms p80; the maintainers' private latency assessment is separate. Longer requests require additional memory and might exceed the available context or VRAM. Both models run on every request. Model startup loads and quantizes the BF16 checkpoint once, not on each request. The 40,000-character/4 input proxy cap is a preflight limit, not an assurance that all inputs below it fit the model's actual token context.

Tested on Isambard with Python 3.11, PyTorch 2.11.0+cu130, Transformers 5.14.1, Accelerate 1.15.0, safetensors 0.8.0 and kernels 0.16.0. The Qwen adapter requires the specialist reader from `reax-subnet`, pinned to commit `0c1efcc7abad11c56eefc866334aab4b01ec38ec`. Install a CUDA PyTorch build appropriate to the target system, the listed packages, and the pinned subnet reader in the same environment. The upstream repository has other runtimes and their own dependencies; follow their original documentation when using those.

```bash
git clone https://github.com/reaxlabs/reax-subnet.git
git -C reax-subnet checkout 0c1efcc7abad11c56eefc866334aab4b01ec38ec
export PYTHONPATH="$PWD/reax-subnet/src:$PYTHONPATH"
python -m pip install kernels==0.16.0
export LOCAL_KERNELS="kernels-community/finegrained-fp8=$PWD/reax13/third_party/finegrained-fp8"

# Download the private model separately and mount its root as /model.
# It must contain /model/part-g and /model/part-q.
python reax13/copernicus_serve.py --model /model --host 127.0.0.1 --port 8080
```

For offline operation set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. The server provides `GET /healthz` and `POST /v1/systemone` and accepts `model: "copernicus-57b"` when a model field is supplied. The single GPU must be visible as `cuda:0`. An optional two-GPU BF16 reference mode can be enabled in a separate config by setting `precision` to `bf16` and `devices` to `["cuda:0", "cuda:1"]`.

```bash
curl -s http://127.0.0.1:8080/v1/systemone -H 'content-type: application/json' \
  -d '{"model":"copernicus-57b","state":"A parcel arrived late.","questions":{"route":{"type":"choice","instructions":"Which team should handle the complaint?","criteria":{"delivery":"Delivery support","billing":"Billing support"}}}}'
```

`copernicus_engine.CopernicusEngine` also exposes `engine(state, questions) -> (response, diagnostics)` for a Decision Index adapter. Use the pinned Decision Index kit and validate the complete official contract before submitting results. The model requires native model licences and third-party notices; REAX Engine's MIT licence does not relicense model weights.
