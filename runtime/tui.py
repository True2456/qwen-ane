# SPDX-License-Identifier: Apache-2.0
"""runtime/tui.py - Live Terminal User Interface for Rindi Hybrid Engine."""

from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional

# ANSI Color and Formatting Constants
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_RED = "\033[31m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_BLUE = "\033[34m"
C_MAGENTA = "\033[35m"
C_CYAN = "\033[36m"
C_WHITE = "\033[37m"


class RindiTUI:
    """Thread-safe Live Terminal Dashboard and Interactive Controller for Rindi."""

    def __init__(self, engine: Any, host: str = "127.0.0.1", port: int = 2456, max_logs: int = 100):
        self.engine = engine
        self.host = host
        self.port = port
        self.logs: Deque[str] = deque(maxlen=max_logs)
        self.lock = threading.Lock()
        self.running = True
        self.last_ttft_ms = 0.0
        self.last_decode_tps = 0.0
        self.last_tokens_saved = 0
        self.last_apc_hit = False
        self.active_request: Optional[str] = None
        self.test_dispatch_fn: Optional[Callable] = None

    def log(self, message: str, tag: str = "INFO"):
        """Log a formatted message with timestamp."""
        ts = time.strftime("%H:%M:%S")
        tag_color = {
            "INFO": f"{C_CYAN}[INFO]{C_RESET}",
            "HTTP": f"{C_BLUE}[HTTP]{C_RESET}",
            "APC": f"{C_GREEN}[APC]{C_RESET}",
            "TOOL": f"{C_MAGENTA}[TOOL]{C_RESET}",
            "TURBO": f"{C_YELLOW}[TURBO]{C_RESET}",
            "SILENT": f"{C_GREEN}[SILENT]{C_RESET}",
            "ERROR": f"{C_RED}[ERROR]{C_RESET}",
        }.get(tag, f"[{tag}]")

        formatted = f"{C_DIM}{ts}{C_RESET} {tag_color} {message}"
        with self.lock:
            self.logs.append(formatted)
        print(formatted, flush=True)

    def record_metrics(self, ttft_ms: float, decode_tps: float, apc_hit: bool, tokens_saved: int):
        """Record latest generation statistics."""
        with self.lock:
            self.last_ttft_ms = ttft_ms
            self.last_decode_tps = decode_tps
            self.last_apc_hit = apc_hit
            self.last_tokens_saved = tokens_saved

    def render_dashboard(self) -> str:
        """Render the full ANSI dashboard."""
        st = self.engine.apc.stats()
        mode_str = (
            f"{C_BOLD}{C_YELLOW}⚡ TURBO (MTP+ANE){C_RESET}"
            if self.engine.mode == "turbo"
            else f"{C_BOLD}{C_GREEN}🌿 SILENT (Pure ANE @ ~5.9W){C_RESET}"
        )

        apc_color = C_GREEN if st["hit_rate_pct"] > 50 else (C_YELLOW if st["hit_rate_pct"] > 0 else C_DIM)

        lines = [
            f"{C_BOLD}{C_CYAN}╔══════════════════════════════════════════════════════════════════════════════════════════════╗{C_RESET}",
            f"{C_BOLD}{C_CYAN}║{C_RESET}  {C_BOLD}RINDI HYBRID INFERENCE ENGINE{C_RESET} │ Apple M5 Max (Metal + 64 ANE Resident Layers)           {C_BOLD}{C_CYAN}║{C_RESET}",
            f"{C_BOLD}{C_CYAN}╠══════════════════════════════════════════════════════════════════════════════════════════════╣{C_RESET}",
            f"{C_BOLD}{C_CYAN}║{C_RESET}  Endpoint: {C_BOLD}http://{self.host}:{self.port}/v1{C_RESET}  │ Model: {C_BOLD}Qwen3.8-27B{C_RESET}  │ Mode: {mode_str:<32} {C_BOLD}{C_CYAN}║{C_RESET}",
            f"{C_BOLD}{C_CYAN}║{C_RESET}  Power: {C_GREEN}~5.9 W SoC{C_RESET}  │ Memory: {C_MAGENTA}12.19 GB ANE blobs{C_RESET} (41.0 GB Host RAM freed)            {C_BOLD}{C_CYAN}║{C_RESET}",
            f"{C_BOLD}{C_CYAN}╠══════════════════════════════════════════════════════════════════════════════════════════════╣{C_RESET}",
            f"{C_BOLD}{C_CYAN}║{C_RESET}  {C_BOLD}PERFORMANCE & METRICS{C_RESET}                                                                {C_BOLD}{C_CYAN}║{C_RESET}",
            f"{C_BOLD}{C_CYAN}║{C_RESET}  • TTFT: {C_BOLD}{self.last_ttft_ms:.2f} ms{C_RESET} {'(APC Hit: 0 FLOPs)' if self.last_apc_hit else '(Prefill)'} │ Decode Speed: {C_BOLD}{self.last_decode_tps:.1f} tok/s{C_RESET}                {C_BOLD}{C_CYAN}║{C_RESET}",
            f"{C_BOLD}{C_CYAN}║{C_RESET}  • APC Prefix Cache: {apc_color}{st['hit_rate_pct']:.1f}% hit rate{C_RESET} ({st['hits']}/{st['total_requests']} reqs) │ Saved: {C_GREEN}{st['tokens_saved']:,} tokens{C_RESET} ({st['total_tokens_stored']:,} cached)  {C_BOLD}{C_CYAN}║{C_RESET}",
            f"{C_BOLD}{C_CYAN}╠══════════════════════════════════════════════════════════════════════════════════════════════╣{C_RESET}",
            f"{C_BOLD}{C_CYAN}║{C_RESET}  {C_DIM}Commands: [t]urbo │ [s]ilent │ [c]lear-cache │ [stats] │ /chat <msg> │ [q]uit{C_RESET}             {C_BOLD}{C_CYAN}║{C_RESET}",
            f"{C_BOLD}{C_CYAN}╚══════════════════════════════════════════════════════════════════════════════════════════════╝{C_RESET}",
        ]
        return "\n".join(lines)

    def print_status(self):
        """Print clean status overview."""
        print(self.render_dashboard(), flush=True)

    def run_interactive_loop(self, dispatch_fn: Optional[Callable] = None):
        """Interactive input loop."""
        self.test_dispatch_fn = dispatch_fn
        while self.running:
            try:
                line = sys.stdin.readline()
                if not line:
                    time.sleep(1.0)
                    continue
                cmd = line.strip()
                if not cmd:
                    continue

                if cmd.lower() in ("t", "turbo"):
                    self.engine.mode = "turbo"
                    self.log(f"Switched to TURBO MODE (Metal GPU MTP + ANE Verifier)", tag="TURBO")
                    self.print_status()
                elif cmd.lower() in ("s", "silent"):
                    self.engine.mode = "silent"
                    self.log(f"Switched to SILENT MODE (Pure ANE @ ~5.9W)", tag="SILENT")
                    self.print_status()
                elif cmd.lower() in ("stats", "status"):
                    self.print_status()
                elif cmd.lower() in ("c", "clear", "clear-cache"):
                    self.engine.apc.root.children.clear()
                    self.engine.apc.total_tokens_stored = 0
                    self.log("APC Prefix Cache cleared (0 tokens cached)", tag="APC")
                    self.print_status()
                elif cmd.startswith("/chat ") or cmd.startswith("chat "):
                    prompt_text = cmd.split(" ", 1)[1]
                    self.log(f"Running interactive test prompt: '{prompt_text}'", tag="INFO")
                    if self.test_dispatch_fn:
                        t0 = time.perf_counter()
                        res = self.test_dispatch_fn(prompt_text, max_tokens=64, mode=self.engine.mode, use_apc=True)
                        t_tot = time.perf_counter() - t0
                        print(f"\n{C_BOLD}{C_GREEN}Assistant:{C_RESET} {res.get('generated_text', '')}\n")
                        self.log(f"Generated {res.get('generated_tokens', 0)} tokens in {t_tot:.2f}s ({res.get('decode_tps', 0):.1f} tok/s)", tag="INFO")
                elif cmd.lower() in ("q", "quit", "exit"):
                    self.log("Shutting down Rindi server...", tag="INFO")
                    self.running = False
                    os._exit(0)
                elif cmd.lower() in ("h", "help", "?"):
                    print(f"\n{C_BOLD}Rindi Interactive Commands:{C_RESET}")
                    print(f"  t / turbo       - Switch to Turbo Mode (Metal GPU MTP + ANE)")
                    print(f"  s / silent      - Switch to Silent Mode (Pure ANE @ ~5.9W)")
                    print(f"  stats / status  - Refresh dashboard & performance metrics")
                    print(f"  clear           - Flush the APC Radix Prefix Cache")
                    print(f"  /chat <prompt>  - Test prompt directly inside the server TUI")
                    print(f"  q / quit        - Exit server\n")
            except (EOFError, KeyboardInterrupt):
                time.sleep(1.0)
            except Exception as e:
                self.log(f"Interactive command error: {e}", tag="ERROR")
