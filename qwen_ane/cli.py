"""Master CLI command line parser and dispatcher for qwen-ane."""
from __future__ import annotations

import argparse
import sys
from typing import Any, Sequence

from .config import load_config, save_config, get_qwen_ane_dir
from .downloader import ensure_model, find_local_model, build_from_base, normalize_model_name
from .server import run_server
from .chat import start_chat


def parse_context_size(val: str | int) -> int:
    """Parse context size strings like '128k', '64k', '8192', '4096'."""
    s = str(val).strip().lower()
    if s.endswith("k"):
        try:
            return int(float(s[:-1]) * 1024)
        except ValueError:
            pass
    try:
        return int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid context size: {val!r}")


class FlexibleArgumentParser(argparse.ArgumentParser):
    """Argument parser that accepts both single-dash (-ctx) and double-dash (--ctx) flags."""

    def _get_option_tuples(self, option_string: str):
        # Allow single-dash long options like -ctx, -model, -port, -lru
        return super()._get_option_tuples(option_string)


def create_parser() -> argparse.ArgumentParser:
    cfg = load_config()

    parser = argparse.ArgumentParser(
        prog="qwen-ane",
        description="qwen-ane: High-performance Qwen inference on Apple Silicon Neural Engine (ANE)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common model/runtime options
    def add_common_args(p: argparse.ArgumentParser):
        p.add_argument(
            "-m", "--model", "-model",
            default=cfg.get("default_model", "flash-next"),
            help="Model name ('flash-next' or '27b', default: flash-next)",
        )
        p.add_argument(
            "-c", "--ctx", "-ctx",
            type=parse_context_size,
            default=None,
            help="Context window size (e.g. 128k, 64k, 32k, 8192, 4096. Up to 256k)",
        )
        p.add_argument(
            "-p", "--port", "-port",
            type=int,
            default=None,
            help="Port to listen on or connect to (default: 2457 for flash-next, 1240 for 27b)",
        )
        p.add_argument(
            "--host", "-host",
            default=cfg.get("default_host", "127.0.0.1"),
            help="Host address (default: 127.0.0.1)",
        )
        p.add_argument(
            "-lru", "--lru",
            dest="lru",
            action="store_true",
            default=cfg.get("default_lru", True),
            help="Enable recurrent and KV state prefix caching (default: enabled)",
        )
        p.add_argument(
            "--no-lru", "--no-cache",
            dest="lru",
            action="store_false",
            help="Disable state prefix caching",
        )
        p.add_argument(
            "--model-path", "-model-path",
            default=None,
            help="Explicit filesystem path to model weights",
        )
        p.add_argument(
            "--hf-repo", "-hf-repo",
            default=None,
            help="Custom Hugging Face repository ID to pull weights from",
        )

    # 1. 'chat' command
    p_chat = subparsers.add_parser(
        "chat",
        help="Start an interactive chat session (Apple Foundation Models 'fm' style)",
    )
    add_common_args(p_chat)
    p_chat.add_argument(
        "-i", "--instructions", "-instructions", "--system", "-system",
        default=None,
        help="System instructions for the model",
    )
    p_chat.add_argument(
        "--thinking", "-thinking",
        choices=["off", "low", "medium", "xhigh"],
        default=cfg.get("default_thinking", "off"),
        help="Thinking mode reasoning effort (default: off)",
    )
    p_chat.add_argument(
        "-r", "--resume", "-resume",
        default=None,
        help="Resume a previous chat session ID",
    )
    p_chat.add_argument(
        "--temp", "--temperature", "-temp",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    p_chat.add_argument(
        "--max-tokens", "-max-tokens", "--max-new", "-max-new",
        type=int,
        default=2048,
        help="Maximum tokens to generate per turn (default: 2048)",
    )

    # 2. 'serve' command
    p_serve = subparsers.add_parser(
        "serve",
        help="Start an OpenAI-compatible Chat Completions API server",
    )
    add_common_args(p_serve)
    p_serve.add_argument(
        "--max-new", "-max-new", "--max-tokens", "-max-tokens",
        type=int,
        default=2048,
        help="Maximum completion tokens cap (default: 2048)",
    )
    p_serve.add_argument(
        "--spec", "-spec",
        type=int,
        default=4,
        help="Speculative decoding lookahead steps (default: 4)",
    )
    p_serve.add_argument(
        "--mtp-draft", "-mtp-draft", "--mtp", "-mtp",
        type=int,
        default=None,
        choices=[0, 1, 2, 3],
        help="MTP speculative draft depth for 27b (0-3, default: 2 when supported, else 0)",
    )

    # 3. 'pull' / 'download' command
    p_pull = subparsers.add_parser(
        "pull",
        help="Download model weights from Hugging Face into .qwenANE/models/",
    )
    p_pull.add_argument("model", help="Model name ('flash-next' or '27b')")
    p_pull.add_argument("--hf-repo", "-hf-repo", default=None, help="Custom HF repository ID")

    # 4. 'build' command
    p_build = subparsers.add_parser(
        "build",
        help="Build and quantize ANE weights from base BF16 weights",
    )
    p_build.add_argument("model", help="Model name ('flash-next' or '27b')")
    p_build.add_argument("--source", "-source", "-s", required=True, help="Path to base BF16 weights")
    p_build.add_argument("--out-dir", "-out-dir", "-o", default=None, help="Custom output directory")

    # 5. 'models' command
    p_models = subparsers.add_parser(
        "models",
        help="List supported models, local cache status, and hardware requirements",
    )

    # 6. 'config' command
    p_config = subparsers.add_parser("config", help="View or modify user configuration")
    p_config.add_argument("action", choices=["show", "get", "set"], default="show", nargs="?")
    p_config.add_argument("key", nargs="?", help="Configuration key")
    p_config.add_argument("value", nargs="?", help="Configuration value")

    # 6. 'bench' command
    p_bench = subparsers.add_parser(
        "bench",
        help="Run power, prefill/decode speed, and context scaling benchmark (with powermetrics)",
    )
    p_bench.add_argument(
        "-m", "--model", "-model",
        default="flash-next",
        help="Model to benchmark ('flash-next', '27b', or '27b,flash-next', default: flash-next)",
    )
    p_bench.add_argument(
        "--pp",
        default=None,
        help="Comma-separated prompt token lengths (default: 1024,2048,4096 for 27B; 4096,8192,16384,32768 for Flash-Next)",
    )
    p_bench.add_argument(
        "--tg",
        type=int,
        default=128,
        help="Number of tokens to generate (default: 128)",
    )
    p_bench.add_argument(
        "-p", "--port", "-port",
        type=int,
        default=1240,
        help="Port for 27B benchmark server (default: 1240)",
    )
    p_bench.add_argument(
        "--skip-power",
        action="store_true",
        help="Skip powermetrics sampling",
    )
    p_bench.add_argument(
        "--model-path", "-model-path",
        default=None,
        help="Explicit path to model directory",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Pre-normalize single-dash long options like -ctx, -model, -port, -lru, -thinking
    normalized_argv = []
    known_long_flags = {
        "-ctx": "--ctx",
        "-model": "--model",
        "-port": "--port",
        "-host": "--host",
        "-lru": "--lru",
        "-no-lru": "--no-lru",
        "-no-cache": "--no-cache",
        "-thinking": "--thinking",
        "-instructions": "--instructions",
        "-system": "--system",
        "-resume": "--resume",
        "-temp": "--temp",
        "-max-tokens": "--max-tokens",
        "-max-new": "--max-new",
        "-model-path": "--model-path",
        "-hf-repo": "--hf-repo",
        "-source": "--source",
        "-out-dir": "--out-dir",
        "-spec": "--spec",
    }
    for arg in argv:
        if arg in known_long_flags:
            normalized_argv.append(known_long_flags[arg])
        elif "=" in arg and arg.split("=")[0] in known_long_flags:
            k, v = arg.split("=", 1)
            normalized_argv.append(f"{known_long_flags[k]}={v}")
        else:
            normalized_argv.append(arg)

    # Default to 'chat' if no command provided
    if not normalized_argv or (normalized_argv and normalized_argv[0].startswith("-")):
        normalized_argv = ["chat"] + normalized_argv

    parser = create_parser()
    args = parser.parse_args(normalized_argv)

    if args.command == "serve":
        return run_server(
            model=args.model,
            ctx=args.ctx,
            port=args.port,
            host=args.host,
            lru=args.lru,
            model_path=args.model_path,
            hf_repo=args.hf_repo,
            max_new=args.max_new,
            spec=args.spec,
            mtp_draft=getattr(args, "mtp_draft", None),
        )

    elif args.command == "chat":
        return start_chat(
            model=args.model,
            ctx=args.ctx,
            port=args.port,
            host=args.host,
            lru=args.lru,
            thinking=args.thinking,
            system_prompt=args.instructions,
            resume_id=args.resume,
            temperature=args.temp,
            max_tokens=args.max_tokens,
            model_path=args.model_path,
            hf_repo=args.hf_repo,
        )

    elif args.command == "pull":
        path = ensure_model(args.model, hf_repo=args.hf_repo)
        print(f"✅ Model ready at: {path}")
        return 0

    elif args.command == "build":
        path = build_from_base(args.model, source_path=args.source, output_dir=args.out_dir)
        print(f"✅ Model built at: {path}")
        return 0

    elif args.command == "models":
        models = [
            {
                "id": "flash-next",
                "name": "Qwen3.8-Flash-Next",
                "hardware": "Apple Neural Engine (GDN/QSA) + MLX (MoE)",
                "context": "Up to 128k tokens",
                "memory": "~23 GB unified memory",
            },
            {
                "id": "27b",
                "name": "Qwen3.8-27B (Pure ANE)",
                "hardware": "Apple Neural Engine (100% pure on-chip)",
                "context": "Up to 4096 tokens (hardware tiled)",
                "memory": "~13 GB resident ANE memory",
            },
        ]
        print("\nSupported Models in qwen-ane:\n")
        for m in models:
            local = find_local_model(m["id"])
            status = f"✅ Available locally ({local})" if local else "☁️  Not downloaded (will pull on first use)"
            print(f"  • {m['id']} ({m['name']})")
            print(f"    Architecture: {m['hardware']}")
            print(f"    Context:      {m['context']}")
            print(f"    RAM Footprint:{m['memory']}")
            print(f"    Status:       {status}\n")
        return 0

    elif args.command == "config":
        cfg = load_config()
        if args.action in ("show", None) and not args.key:
            print(f"\nConfiguration ({get_qwen_ane_dir() / 'config.json'}):\n")
            for k, v in cfg.items():
                print(f"  {k}: {v}")
            print()
            return 0
        elif args.action == "get" or (args.action == "show" and args.key):
            val = cfg.get(args.key)
            print(val if val is not None else "")
            return 0
        elif args.action == "set":
            if not args.key or args.value is None:
                print("Usage: qwen-ane config set <key> <value>")
                return 1
            # Auto-cast bools/ints
            val: Any = args.value
            if val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False
            elif val.isdigit():
                val = int(val)
            cfg[args.key] = val
            save_config(cfg)
            print(f"Updated {args.key} = {val}")
            return 0
    elif args.command == "bench":
        import probes.ctx_scale_bench as csb
        model_canon = "27b" if "27b" in args.model.lower() else "flash-next"
        default_pp = "1024,2048,4096" if model_canon == "27b" else "4096,8192,16384,32768"
        pp_val = args.pp if args.pp else default_pp
        bench_args = [
            f"--models={args.model}",
            f"--pp={pp_val}",
            f"--tg={args.tg}",
            f"--port={args.port}",
        ]
        if args.skip_power:
            bench_args.append("--skip-power")
        if args.model_path:
            if "27b" in args.model:
                bench_args.append(f"--model-27b={args.model_path}")
            if "flash" in args.model:
                bench_args.append(f"--model-flashnext={args.model_path}")
        old_argv = sys.argv
        try:
            sys.argv = ["ctx_scale_bench.py", *bench_args]
            return csb.main()
        finally:
            sys.argv = old_argv

    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
