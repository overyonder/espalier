"""Generate or edit images through the ChatGPT plan's Codex images endpoint.

  chatgpt_image.py PROMPT_FILE OUT.png [IMAGE ...] [--size WxH] [--quality high|medium|low] [--model NAME]

The endpoint does not report which model ran and may serve a lower quality than requested.
"""
import argparse
import base64
import json
import math
import mimetypes
import random
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request
from email.utils import parsedate_to_datetime
from http.client import IncompleteRead
from pathlib import Path

ENDPOINT = "https://chatgpt.com/backend-api/codex/images/"
REQUEST_INTERVAL = 10.0
MAX_RETRIES = 5


class CodexImageClient:
    """Serial image requests with pacing and bounded transient-error retries."""

    def __init__(self):
        import os
        auth_paths = ([Path(os.environ["CODEX_AUTH_PATH"])] if os.environ.get("CODEX_AUTH_PATH")
                      else [Path.home() / ".codex/auth.json", Path.home() / ".config/codex/auth.json"])
        auth_path = next((path for path in auth_paths if path.is_file()), None)
        if auth_path is None:
            raise RuntimeError("Codex login not found; run codex login")
        tokens = json.loads(auth_path.read_text())["tokens"]
        self.headers = {
            "Authorization": "Bearer " + tokens["access_token"],
            "chatgpt-account-id": tokens["account_id"],
            "originator": "codex_cli_rs",
            "Content-Type": "application/json",
            "User-Agent": "codex_cli_rs/0.159.3",
        }
        system_ca = Path("/etc/ssl/certs/ca-certificates.crt")
        self.ssl_context = ssl.create_default_context(
            cafile=os.environ.get("SSL_CERT_FILE") or (str(system_ca) if system_ca.is_file() else None)
        )
        self.next_request_at = 0.0

    def generate(self, prompt, output, images=(), size="2048x1152",
                 quality="high", model="gpt-image-2.5-sunburst"):
        body = {"prompt": prompt, "model": model, "quality": quality, "size": size}
        if images:
            body["images"] = []
            for image in images:
                path = Path(image)
                mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                body["images"].append({"image_url": f"data:{mime};base64,{encoded}"})
        request = urllib.request.Request(
            ENDPOINT + ("edits" if images else "generations"),
            data=json.dumps(body).encode(), headers=self.headers,
        )
        started = time.monotonic()
        for attempt in range(MAX_RETRIES + 1):
            delay = max(0.0, self.next_request_at - time.monotonic())
            if delay:
                print(f"GPT request waiting {delay:.1f}s", file=sys.stderr, flush=True)
                time.sleep(delay)
            self.next_request_at = time.monotonic() + REQUEST_INTERVAL
            retry_after = None
            try:
                with urllib.request.urlopen(request, timeout=900, context=self.ssl_context) as response:
                    data = json.load(response)
                break
            except urllib.error.HTTPError as error:
                error_body = error.read()
                error.close()
                try:
                    detail = json.loads(error_body).get("error", {})
                    code = detail.get("code") if isinstance(detail, dict) else None
                except (ValueError, AttributeError):
                    code = None
                retryable = error.code in (408, 409, 429) or 500 <= error.code <= 599
                if (not retryable or code in ("insufficient_quota", "organization_usage_limit_exceeded")
                        or attempt == MAX_RETRIES):
                    raise RuntimeError(f"GPT image request failed: HTTP {error.code}") from None
                value = error.headers.get("Retry-After")
                if value is not None:
                    try:
                        retry_after = float(value)
                    except ValueError:
                        try:
                            retry_after = parsedate_to_datetime(value).timestamp() - time.time()
                        except (ValueError, TypeError, OverflowError):
                            pass
                failure = f"HTTP {error.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, IncompleteRead) as error:
                if attempt == MAX_RETRIES:
                    raise RuntimeError("GPT image request failed after connection retries") from error
                failure = f"connection failure ({type(getattr(error, 'reason', error)).__name__})"
            if retry_after is None or not math.isfinite(retry_after):
                retry_after = min(300.0, REQUEST_INTERVAL * 2 ** attempt) + random.uniform(0, 1)
            retry_after = max(REQUEST_INTERVAL, retry_after)
            self.next_request_at = max(self.next_request_at, time.monotonic() + retry_after)
            print(f"GPT {failure}; retry {attempt + 1}/{MAX_RETRIES} in {retry_after:.1f}s",
                  file=sys.stderr, flush=True)

        image_bytes = base64.b64decode(data["data"][0]["b64_json"], validate=True)
        if not image_bytes:
            raise RuntimeError("GPT returned an empty image")
        output = Path(output)
        pending = None
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix=output.name + ".", delete=False) as stream:
                pending = Path(stream.name)
                stream.write(image_bytes)
            pending.replace(output)
        finally:
            if pending is not None:
                pending.unlink(missing_ok=True)
        print(output, f"{time.monotonic() - started:.0f}s", data.get("quality"), data.get("size"), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("out")
    parser.add_argument("images", nargs="*")
    parser.add_argument("--size", default="2048x1152")
    parser.add_argument("--quality", default="high")
    parser.add_argument("--model", default="gpt-image-2.5-sunburst")
    args = parser.parse_args()
    try:
        CodexImageClient().generate(Path(args.prompt).read_text(), args.out, args.images,
                                    args.size, args.quality, args.model)
    except (RuntimeError, urllib.error.URLError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
