import dataclasses
import json
import queue
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ReadOnly, TypedDict, TypeIs, cast, get_args

from baby_whale_v4.config import BabyWhaleV4Config
from baby_whale_v4.data.chat import render_chat_prompt
from baby_whale_v4.data.tokenizer import ByteTokenizer, Tokenizer
from baby_whale_v4.inference.engine import Engine, GenerationOptions
from baby_whale_v4.inference.prefix_cache import PrefixCache
from baby_whale_v4.inference.serving import BatchingServer
from baby_whale_v4.model import BabyWhaleV4Model
from baby_whale_v4.typing import GenerationMode

_GENERATION_MODES: tuple[GenerationMode, ...] = get_args(GenerationMode)
_GENERATE_KEYS = frozenset({"prompt", "max_new_tokens", "mode"})
_CHAT_COMPLETIONS_KEYS = frozenset(
    {"model", "messages", "max_tokens", "max_new_tokens", "mode", "stream"}
)
_ROLLOUT_KEYS = frozenset({"prompt_ids", "options"})
_ROLLOUT_OPTION_KEYS = frozenset({"max_new_tokens", "mode", "temperature", "top_k", "eos_id"})
_SYNC_WEIGHTS_KEYS = frozenset({"checkpoint_path"})


class _GenerateRequest(TypedDict, total=False):
    prompt: ReadOnly[str]
    max_new_tokens: ReadOnly[int]
    mode: ReadOnly[GenerationMode]


class _ChatCompletionsRequest(TypedDict, total=False):
    model: ReadOnly[str]
    messages: ReadOnly[list[dict[str, str]]]
    max_tokens: ReadOnly[int]
    max_new_tokens: ReadOnly[int]
    mode: ReadOnly[GenerationMode]
    stream: ReadOnly[bool]


def _is_generation_mode(value: object) -> TypeIs[GenerationMode]:
    return isinstance(value, str) and value in _GENERATION_MODES


def _parse_generate_request(raw: object) -> _GenerateRequest:
    if not isinstance(raw, dict):
        raise ValueError("JSON body must be an object")
    if not all(isinstance(key, str) for key in raw):
        raise ValueError("JSON object keys must be strings")
    raw_dict = cast(dict[str, object], raw)
    unknown = set(raw_dict) - _GENERATE_KEYS
    if unknown:
        raise ValueError(f"unknown JSON keys: {sorted(unknown)}")
    if "prompt" not in raw_dict:
        raise ValueError("prompt is required")

    payload: dict[str, object] = {}
    prompt = raw_dict["prompt"]
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    if not prompt:
        raise ValueError("prompt must be non-empty")
    payload["prompt"] = prompt
    if "max_new_tokens" in raw_dict:
        max_new_tokens = raw_dict["max_new_tokens"]
        if type(max_new_tokens) is not int:
            raise ValueError("max_new_tokens must be an integer")
        payload["max_new_tokens"] = max_new_tokens
    if "mode" in raw_dict:
        mode = raw_dict["mode"]
        if not _is_generation_mode(mode):
            raise ValueError(f"mode must be one of: {list(_GENERATION_MODES)}")
        payload["mode"] = mode
    return cast(_GenerateRequest, payload)


def _parse_chat_completions_request(raw: object) -> _ChatCompletionsRequest:
    if not isinstance(raw, dict):
        raise ValueError("JSON body must be an object")
    if not all(isinstance(key, str) for key in raw):
        raise ValueError("JSON object keys must be strings")
    raw_dict = cast(dict[str, object], raw)
    unknown = set(raw_dict) - _CHAT_COMPLETIONS_KEYS
    if unknown:
        raise ValueError(f"unknown JSON keys: {sorted(unknown)}")
    if "messages" not in raw_dict:
        raise ValueError("messages is required")
    payload: dict[str, object] = {}
    model = raw_dict.get("model", "baby-whale-v4")
    if not isinstance(model, str) or not model:
        raise ValueError("model must be a non-empty string")
    payload["model"] = model
    messages = raw_dict["messages"]
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    parsed_messages: list[dict[str, str]] = []
    for idx, item in enumerate(messages):
        if not isinstance(item, dict) or not all(isinstance(key, str) for key in item):
            raise ValueError(f"messages[{idx}] must be an object with string keys")
        item_dict = cast(dict[str, object], item)
        role = item_dict.get("role")
        content = item_dict.get("content")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"messages[{idx}].role is unsupported")
        if not isinstance(content, str):
            raise ValueError(f"messages[{idx}].content must be a string")
        parsed_messages.append({"role": cast(str, role), "content": content})
    payload["messages"] = parsed_messages
    for key in ("max_tokens", "max_new_tokens"):
        if key in raw_dict:
            value = raw_dict[key]
            if type(value) is not int:
                raise ValueError(f"{key} must be an integer")
            payload[key] = value
    if "stream" in raw_dict:
        stream = raw_dict["stream"]
        if type(stream) is not bool:
            raise ValueError("stream must be a boolean")
        payload["stream"] = stream
    if "mode" in raw_dict:
        mode = raw_dict["mode"]
        if not _is_generation_mode(mode):
            raise ValueError(f"mode must be one of: {list(_GENERATION_MODES)}")
        payload["mode"] = mode
    return cast(_ChatCompletionsRequest, payload)


def _parse_rollout_request(raw: object) -> tuple[list[int], GenerationOptions]:
    if not isinstance(raw, dict):
        raise ValueError("JSON body must be an object")
    if not all(isinstance(key, str) for key in raw):
        raise ValueError("JSON object keys must be strings")
    raw_dict = cast(dict[str, object], raw)
    unknown = set(raw_dict) - _ROLLOUT_KEYS
    if unknown:
        raise ValueError(f"unknown JSON keys: {sorted(unknown)}")
    prompt_ids = raw_dict.get("prompt_ids")
    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise ValueError("prompt_ids must be a non-empty list")
    if not all(type(t) is int for t in prompt_ids):
        raise ValueError("prompt_ids must be a list of integers")
    options_dict = raw_dict.get("options")
    if not isinstance(options_dict, dict) or not all(isinstance(k, str) for k in options_dict):
        raise ValueError("options must be a JSON object with string keys")
    options_typed = cast(dict[str, object], options_dict)
    unknown_opts = set(options_typed) - _ROLLOUT_OPTION_KEYS
    if unknown_opts:
        raise ValueError(f"unknown rollout option keys: {sorted(unknown_opts)}")
    # Narrow each option into a typed local so the final constructor call is
    # fully typed (no dict[str, object] flowing through GenerationOptions(**)).
    opts = GenerationOptions()
    if "max_new_tokens" in options_typed:
        raw_mnt = options_typed["max_new_tokens"]
        if type(raw_mnt) is not int:
            raise ValueError("max_new_tokens must be an integer")
        opts = dataclasses.replace(opts, max_new_tokens=raw_mnt)
    if "mode" in options_typed:
        raw_mode = options_typed["mode"]
        if not _is_generation_mode(raw_mode):
            raise ValueError(f"mode must be one of: {list(_GENERATION_MODES)}")
        opts = dataclasses.replace(opts, mode=raw_mode)
    if "temperature" in options_typed:
        raw_temp = options_typed["temperature"]
        if type(raw_temp) not in (int, float):
            raise ValueError("temperature must be a number")
        opts = dataclasses.replace(opts, temperature=float(cast(int | float, raw_temp)))
    if "top_k" in options_typed:
        raw_tk = options_typed["top_k"]
        if raw_tk is not None and type(raw_tk) is not int:
            raise ValueError("top_k must be an integer or null")
        opts = dataclasses.replace(opts, top_k=raw_tk)
    if "eos_id" in options_typed:
        raw_eos = options_typed["eos_id"]
        if raw_eos is not None and type(raw_eos) is not int:
            raise ValueError("eos_id must be an integer or null")
        opts = dataclasses.replace(opts, eos_id=raw_eos)
    return cast(list[int], prompt_ids), opts


_REQUEST_TIMEOUT = 120.0  # seconds a handler waits on the serving loop


def _chat_finish_reason(generated: list[int], eos_id: int | None) -> str:
    """OpenAI ``finish_reason``: 'stop' when the last token is EOS, else 'length'."""
    if eos_id is not None and generated and generated[-1] == eos_id:
        return "stop"
    return "length"


@dataclass
class ServeContext:
    engine: Engine
    tokenizer: Tokenizer
    config: BabyWhaleV4Config
    model: BabyWhaleV4Model | None = None
    prefill_chunk: int = 4
    batcher: BatchingServer = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.engine, Engine):
            raise TypeError("ServeContext.engine must be an Engine")
        if not isinstance(self.config, BabyWhaleV4Config):
            raise TypeError("ServeContext.config must be a BabyWhaleV4Config")
        if self.model is None:
            self.model = self.engine.model
        self.batcher = BatchingServer(self.engine, prefill_chunk=self.prefill_chunk)

    def start(self) -> None:
        """Start the background continuous-batching loop before serving requests."""
        self.batcher.start()

    def stop(self) -> None:
        self.batcher.stop()


def make_handler(ctx: ServeContext):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # silence stdout
            return

        def do_POST(self) -> None:
            if self.path == "/generate":
                self._handle_generate()
                return
            if self.path == "/v1/chat/completions":
                self._handle_chat_completions()
                return
            if self.path == "/rollout":
                self._handle_rollout()
                return
            if self.path == "/sync_weights":
                self._handle_sync_weights()
                return
            self.send_response(404)
            self.end_headers()

        def _handle_sync_weights(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid JSON"})
                return
            if not isinstance(body, dict) or set(body) - _SYNC_WEIGHTS_KEYS:
                self._send_json(400, {"error": "expected {checkpoint_path: str}"})
                return
            checkpoint_path = body.get("checkpoint_path")
            if not isinstance(checkpoint_path, str) or not checkpoint_path:
                self._send_json(400, {"error": "checkpoint_path must be a non-empty string"})
                return

            def _do_sync(engine: Engine) -> dict[str, object]:
                # Runs on the serving loop thread — exclusive model access, so
                # the weight swap can never race an in-flight forward.
                from baby_whale_v4.training.checkpoint import load_checkpoint

                ckpt = load_checkpoint(
                    checkpoint_path, expected_config_hash=ctx.config.config_hash()
                )
                target = ctx.model
                if target is None:
                    raise RuntimeError("ServeContext.model unavailable")
                target.update(ckpt.model_state)
                if engine.prefix_cache is not None:
                    engine.prefix_cache.clear()
                return {"config_hash": ckpt.config_hash, "step": ckpt.step}

            try:
                result = ctx.batcher.run_control(_do_sync)
            except (FileNotFoundError, ValueError, TypeError) as exc:
                self._send_json(400, {"error": f"sync failed: {exc}"})
                return
            except RuntimeError as exc:
                self._send_json(500, {"error": str(exc)})
                return
            if not isinstance(result, dict):
                self._send_json(500, {"error": "sync produced no result"})
                return
            self._send_json(200, {"loaded": True, **cast(dict[str, object], result)})

        def _handle_rollout(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            try:
                prompt_ids, options = _parse_rollout_request(
                    json.loads(self.rfile.read(length).decode("utf-8"))
                )
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid JSON"})
                return
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            handle = ctx.batcher.submit(prompt_ids, options)
            try:
                state = handle.result(timeout=_REQUEST_TIMEOUT)
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            self._send_json(
                200,
                {
                    "response_ids": list(state.generated),
                    "log_probs": list(state.captured_log_probs),
                    "finished": state.finished,
                },
            )

        def _handle_generate(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = _parse_generate_request(json.loads(self.rfile.read(length).decode("utf-8")))
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid JSON"})
                return
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            prompt = body["prompt"]
            max_new = body.get("max_new_tokens", 32)
            mode = body.get("mode", "greedy")
            try:
                opts = GenerationOptions(max_new_tokens=max_new, mode=mode)
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            prompt_ids = ctx.tokenizer.encode(prompt)
            handle = ctx.batcher.submit(prompt_ids, opts)
            try:
                state = handle.result(timeout=_REQUEST_TIMEOUT)
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            gen_ids = list(state.generated)
            self._send_json(
                200,
                {
                    "completion": ctx.tokenizer.decode(gen_ids),
                    "prompt_tokens": len(prompt_ids),
                    "generated_tokens": len(gen_ids),
                },
            )

        def _handle_chat_completions(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            try:
                body = _parse_chat_completions_request(
                    json.loads(self.rfile.read(length).decode("utf-8"))
                )
            except json.JSONDecodeError:
                self._send_json(400, {"error": "invalid JSON"})
                return
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            max_new = body.get("max_new_tokens", body.get("max_tokens", 32))
            mode = body.get("mode", "greedy")
            try:
                opts = GenerationOptions(
                    max_new_tokens=max_new, mode=mode, eos_id=ctx.tokenizer.eos_id
                )
            except ValueError as e:
                self._send_openai_error(400, str(e))
                return
            prompt = render_chat_prompt(body["messages"])
            prompt_ids = ctx.tokenizer.encode(prompt)
            if body.get("stream", False):
                self._stream_chat(model=body["model"], prompt_ids=prompt_ids, options=opts)
                return
            handle = ctx.batcher.submit(prompt_ids, opts)
            try:
                state = handle.result(timeout=_REQUEST_TIMEOUT)
            except ValueError as e:
                self._send_json(400, {"error": str(e)})
                return
            gen_ids = list(state.generated)
            self._send_json(
                200,
                {
                    "id": "chatcmpl-baby-whale-v4-local",
                    "object": "chat.completion",
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": ctx.tokenizer.decode(gen_ids),
                            },
                            "finish_reason": _chat_finish_reason(gen_ids, opts.eos_id),
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt_ids),
                        "completion_tokens": len(gen_ids),
                        "total_tokens": len(prompt_ids) + len(gen_ids),
                    },
                },
            )

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "model": ctx.config.name,
                        "config_hash": ctx.config.config_hash(),
                    },
                )
                return
            self.send_response(404)
            self.end_headers()

        def _send_openai_error(
            self, code: int, message: str, err_type: str = "invalid_request_error"
        ) -> None:
            # OpenAI-shaped error object for the /v1/chat/completions surface.
            self._send_json(code, {"error": {"message": message, "type": err_type, "code": None}})

        def _send_json(self, code: int, payload: dict[str, object]) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _stream_chat(
            self, *, model: str, prompt_ids: list[int], options: GenerationOptions
        ) -> None:
            """Emit each decoded token as its own SSE chunk (real TTFT).

            Submits to the batching loop and streams tokens as they are produced.
            The first token is peeked before any bytes are sent, so a validation
            error (raised on the loop thread) still maps to a clean 400.
            """
            handle = ctx.batcher.submit(prompt_ids, options)
            try:
                first = handle.next_token(timeout=_REQUEST_TIMEOUT)
            except queue.Empty:
                handle.cancel()
                self._send_openai_error(504, "generation timed out", "server_error")
                return
            if first is None and handle.error is not None:
                self._send_openai_error(400, str(handle.error))
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            gone = False

            def _emit(token_id: int) -> None:
                nonlocal gone
                chunk = {
                    "id": "chatcmpl-baby-whale-v4-local",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": ctx.tokenizer.decode([token_id])},
                            "finish_reason": None,
                        }
                    ],
                }
                try:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                except BrokenPipeError, ConnectionResetError:
                    # Client hung up mid-stream — stop wasting compute on it.
                    gone = True
                    handle.cancel()

            if first is not None:
                _emit(first)
                while not gone:
                    try:
                        token_id = handle.next_token(timeout=_REQUEST_TIMEOUT)
                    except queue.Empty:
                        break
                    if token_id is None:
                        break
                    _emit(token_id)

            if gone:
                return

            state = handle.result(timeout=_REQUEST_TIMEOUT)
            final = {
                "id": "chatcmpl-baby-whale-v4-local",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": _chat_finish_reason(list(state.generated), options.eos_id),
                    }
                ],
            }
            try:
                self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except BrokenPipeError, ConnectionResetError:
                handle.cancel()

    return Handler


def serve(
    *,
    model: BabyWhaleV4Model,
    config: BabyWhaleV4Config,
    tokenizer: Tokenizer | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    prefill_chunk: int = 4,
) -> None:
    tok = tokenizer or ByteTokenizer()
    engine = Engine(
        model=model,
        config=config,
        tokenizer_hash=tok.hash_signature(),
        prefix_cache=PrefixCache(capacity=64),
    )
    ctx = ServeContext(engine=engine, tokenizer=tok, config=config, prefill_chunk=prefill_chunk)
    ctx.start()
    server = ThreadingHTTPServer((host, port), make_handler(ctx))
    try:
        server.serve_forever()
    finally:
        ctx.stop()
