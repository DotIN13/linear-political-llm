import json
from typing import Any

import aiohttp
import modal

# Qwen3-VL needs recent Transformers support.
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.19.0",
        "pillow",  # for image decoding
    )
    .uv_pip_install(  # as of vllm 0.19.0, must install transformers separately to use Gemma 4
        "transformers==5.5.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # faster model transfers
)

MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"

# Optional: pin this to a commit SHA from the HF model repo once you deploy.
# Leaving it as None tracks the repo's default revision.
MODEL_REVISION = None

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

FAST_BOOT = False

app = modal.App("qwen3-vl-8b-instruct-vllm")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 30000


@app.function(
    image=vllm_image,
    gpu=f"H200:{N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    secrets=[modal.Secret.from_name("hf-secret")],
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    import json
    import subprocess

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--served-model-name",
        MODEL_NAME,
        "llm",
        "default",
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--uvicorn-log-level=info",
        "--trust-remote-code",
        "--tensor-parallel-size",
        str(N_GPU),
    ]

    if MODEL_REVISION:
        cmd += ["--revision", MODEL_REVISION]

    # enforce-eager disables Torch compilation and CUDA graph capture.
    # Use this for faster cold starts while debugging; disable it for better throughput.
    cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]

    # Qwen3-VL is multimodal. Keep these limits modest so one request
    # cannot accidentally consume the whole replica.
    cmd += [
        "--limit-mm-per-prompt",
        json.dumps(
            {
                "image": 4,
                "video": 0,
            }
        ),
    ]

    # Optional knobs you may want to tune:
    # cmd += ["--max-model-len", "32768"]
    # cmd += ["--gpu-memory-utilization", "0.90"]
    # cmd += ["--max-num-seqs", "32"]

    print("Starting vLLM with command:")
    print(" ".join(cmd))

    subprocess.Popen(cmd)


@app.local_entrypoint()
async def test(test_timeout=10 * MINUTES, content=None, image_url=None, twice=True):
    url = await serve.get_web_url.aio()

    if content is None:
        content = "Describe this image in one sentence."

    if image_url is None:
        image_url = (
            "https://cdn.britannica.com/61/93061-050-99147DCE/"
            "Statue-of-Liberty-Island-New-York-Bay.jpg"
        )

    messages = [
        {
            "role": "system",
            "content": "You are a concise, accurate vision-language assistant.",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": content},
                {
                    "type": "image_url",
                    "image_url": {"url": image_url},
                },
            ],
        },
    ]

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
            up = resp.status == 200

        assert up, f"Failed health check for server at {url}"
        print(f"Successful health check for server at {url}")

        print(f"Sending multimodal request to {url}:")
        print(json.dumps(messages, indent=2))

        await _send_request(session, "llm", messages)

        if twice:
            text_only_messages = [
                {
                    "role": "system",
                    "content": "You are a concise, accurate assistant.",
                },
                {
                    "role": "user",
                    "content": "Give me three practical use cases for Qwen3-VL.",
                },
            ]

            print(f"\nSending text-only request to {url}:")
            print(json.dumps(text_only_messages, indent=2))

            await _send_request(session, "llm", text_only_messages)


async def _send_request(
    session: aiohttp.ClientSession,
    model: str,
    messages: list,
) -> None:
    payload: dict[str, Any] = {
        "messages": messages,
        "model": model,
        "stream": True,
        "max_tokens": 512,
        "temperature": 0.2,
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    async with session.post(
        "/v1/chat/completions",
        json=payload,
        headers=headers,
    ) as resp:
        async for raw in resp.content:
            resp.raise_for_status()

            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue

            if line.startswith("data: "):
                line = line[len("data: ") :]

            chunk = json.loads(line)
            assert chunk["object"] == "chat.completion.chunk"

            delta = chunk["choices"][0]["delta"]
            content = (
                delta.get("content")
                or delta.get("reasoning")
                or delta.get("reasoning_content")
            )

            if content:
                print(content, end="")
            else:
                print("\n", chunk)

    print()