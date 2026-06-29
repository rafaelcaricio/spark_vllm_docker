# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx>=0.27"]
# ///

import argparse
import time

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Say hello and confirm you are running.",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="step3p7")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0.2,
    }

    started = time.perf_counter()
    response = httpx.post(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        json=payload,
        timeout=args.timeout,
    )
    elapsed = time.perf_counter() - started
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print(response.text)
        raise exc
    data = response.json()

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens") or 0
    tok_s = completion_tokens / elapsed if completion_tokens else 0.0

    print(content)
    print()
    print(f"elapsed_s={elapsed:.2f}")
    print(f"completion_tokens={completion_tokens}")
    print(f"tok_s={tok_s:.2f}")
    print(f"usage={usage}")


if __name__ == "__main__":
    main()
