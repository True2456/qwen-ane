"""Apple Foundation Models (fm)-faithful interactive CLI chat for qwen-ane.

Recreates the signature AFM terminal experience:
  - Apple Intelligence truecolor gradient header
  - Subtitle status bar
  - Bordered input pane with readline history & tab completion
  - Live thinking indicator & dim reasoning stream
  - Streaming markdown rendering with code block borders and word wrapping
  - Subtle turn metric badges
  - AFM-aligned slash commands (/model, /think, /instructions, /sessions, etc.)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

try:
    import termios
    import tty
except ImportError:
    termios = None
    tty = None

try:
    import readline
except ImportError:
    readline = None

from .config import get_qwen_ane_dir, get_sessions_dir, load_config
from .downloader import ensure_model, normalize_model_name
from .server import is_server_running

SLASH_COMMAND_INFO = [
    ("/exit", "Exit the chat session"),
    ("/clear", "Clear conversation history and screen"),
    ("/model", "Switch active model (flash-next, 27b)"),
    ("/think", "Set reasoning effort (off, low, medium, xhigh)"),
    ("/instructions", "Set or view system instructions"),
    ("/save", "Save session transcript to disk"),
    ("/sessions", "List saved chat sessions"),
    ("/resume", "Resume a saved session"),
    ("/info", "Display hardware, context, and endpoint info"),
    ("/help", "Show help card"),
    ("/quit", "Exit the chat session"),
]
SLASH_COMMANDS = [cmd for cmd, _ in SLASH_COMMAND_INFO]


def gradient_text(
    text: str,
    start_rgb: tuple[int, int, int] = (130, 215, 90),
    end_rgb: tuple[int, int, int] = (35, 130, 205),
) -> str:
    """Interpolate truecolor ANSI escape codes across characters in text."""
    n = len(text)
    if n <= 1:
        return f"\033[38;2;{start_rgb[0]};{start_rgb[1]};{start_rgb[2]}m{text}\033[0m"
    res = []
    for i, ch in enumerate(text):
        t = i / (n - 1)
        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * t)
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * t)
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * t)
        res.append(f"\033[38;2;{r};{g};{b}m{ch}")
    res.append("\033[0m")
    return "".join(res)


def format_tokens(n: int) -> str:
    if n >= 1024:
        return f"{n // 1024}k" if n % 1024 == 0 else f"{n / 1024:.1f}k"
    return str(n)


def time_ago(ts: float) -> str:
    diff = max(0.0, time.time() - ts)
    if diff < 60:
        return "just now"
    if diff < 3600:
        return f"{int(diff / 60)}m ago"
    if diff < 86400:
        return f"{int(diff / 3600)}h ago"
    return f"{int(diff / 86400)}d ago"


def get_term_width() -> int:
    """Returns dynamic terminal width with a 2-char margin, minimum 20 columns."""
    cols = shutil.get_terminal_size(fallback=(80, 24)).columns
    return max(20, cols - 2)


class ChatSession:
    def __init__(
        self,
        model: str = "flash-next",
        host: str = "127.0.0.1",
        port: int = 2457,
        system_prompt: str | None = None,
        thinking: str = "off",
        session_id: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        self.model = normalize_model_name(model)
        self.model_id = "Qwen3.8-Flash-Next" if self.model == "flash-next" else "Qwen3.8-27B"
        self.host = host
        self.port = port
        self.thinking = thinking
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session_id = session_id or (time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
        self.messages: list[dict[str, Any]] = []

        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

    def save(self, custom_name: str | None = None) -> Path:
        sdir = get_sessions_dir()
        name = custom_name or self.session_id
        if not name.endswith(".json"):
            path = sdir / f"{name}.json"
        else:
            path = sdir / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": self.session_id,
                "model": self.model,
                "thinking": self.thinking,
                "messages": self.messages,
                "saved_at": time.time(),
            }, f, indent=2)
        return path

    def load(self, session_id_or_name: str) -> bool:
        sdir = get_sessions_dir()
        path = sdir / f"{session_id_or_name}.json"
        if not path.exists():
            path = sdir / session_id_or_name
        if not path.exists():
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.session_id = data.get("session_id", session_id_or_name)
                self.model = data.get("model", self.model)
                self.model_id = "Qwen3.8-Flash-Next" if self.model == "flash-next" else "Qwen3.8-27B"
                self.thinking = data.get("thinking", self.thinking)
                self.messages = data.get("messages", [])
                return True
        except Exception:
            return False


class StreamingMarkdownRenderer:
    """Streams model output live while styling markdown headings, bold, inline code, and code blocks."""

    def __init__(self, width: int | None = None):
        self._custom_width = width
        self.col = 0
        self.in_code = False
        self.fence_buf = ""
        self.code_lang = ""

    @property
    def width(self) -> int:
        if self._custom_width is not None:
            return self._custom_width
        return get_term_width()

    def write(self, chunk: str):
        for ch in chunk:
            if ch == "\n":
                sys.stdout.write("\n")
                self.col = 0
                if self.in_code:
                    sys.stdout.write("\033[38;2;120;120;120m│\033[0m ")
                    self.col = 2
                continue

            # Detect code fence ```
            if ch == "`":
                self.fence_buf += "`"
                if self.fence_buf.endswith("```"):
                    self.fence_buf = ""
                    if not self.in_code:
                        self.in_code = True
                        dash = "─" * max(2, self.width - 2)
                        border = f"\n\033[38;2;120;120;120m╭{dash}╮\033[0m\n\033[38;2;120;120;120m│\033[0m \033[38;2;180;225;255m"
                        sys.stdout.write(border)
                        self.col = 2
                    else:
                        self.in_code = False
                        dash = "─" * max(2, self.width - 2)
                        border = f"\033[0m\n\033[38;2;120;120;120m╰{dash}╯\033[0m\n"
                        sys.stdout.write(border)
                        self.col = 0
                    continue
                continue
            elif self.fence_buf:
                # Flush non-fence backticks
                sys.stdout.write(self.fence_buf)
                self.col += len(self.fence_buf)
                self.fence_buf = ""

            # Word wrapping near terminal boundary
            if self.col >= self.width and ch == " ":
                sys.stdout.write("\n")
                self.col = 0
                if self.in_code:
                    sys.stdout.write("\033[38;2;120;120;120m│\033[0m ")
                    self.col = 2
                continue

            sys.stdout.write(ch)
            self.col += 1

        sys.stdout.flush()

    def finish(self):
        if self.fence_buf:
            sys.stdout.write(self.fence_buf)
            self.fence_buf = ""
        if self.in_code:
            self.in_code = False
            dash = "─" * max(2, self.width - 2)
            sys.stdout.write(f"\033[0m\n\033[38;2;120;120;120m╰{dash}╯\033[0m\n")
        sys.stdout.write("\n")
        sys.stdout.flush()


def stream_chat_completion(
    host: str,
    port: int,
    model_id: str,
    messages: list[dict[str, Any]],
    thinking: str = "off",
    temperature: float = 0.7,
    max_tokens: int = 2048,
):
    """Streams chat completion tokens from server using SSE."""
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if thinking and thinking != "off":
        payload["enable_thinking"] = True
        payload["reasoning_effort"] = thinking
    else:
        payload["enable_thinking"] = False

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )

    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith("data: "):
                raw_chunk = line[6:]
                if raw_chunk == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw_chunk)
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        yield delta
                except json.JSONDecodeError:
                    continue


def print_banner(model_name: str):
    """Prints the exact Apple Foundation Models-style title banner."""
    print()
    print(gradient_text(" Qwen Neural Engine Chat", (130, 215, 90), (35, 130, 205)))
    print(f"\033[38;2;153;153;153m model: {model_name} · /help for help\033[0m")
    print()


def get_model_badge(canon: str) -> str:
    """Returns the AFM model tag displayed right below the input box."""
    if canon == "27b":
        return "27b (Qwen 3.8 27B Pure ANE)"
    else:
        return "flash-next (Qwen 3.8 Flash-Next ANE+MLX)"


def print_help():
    """Prints AFM-style bordered help card."""
    width = get_term_width()
    dash = "─" * max(2, width - 2)
    header_dash = "─" * max(2, width - 13)
    card = f"""\033[38;2;136;136;136m╭─ Commands {header_dash}╮\033[0m
\033[38;2;136;136;136m│\033[0m  \033[1m/exit, /quit\033[0m          Exit the chat session
\033[38;2;136;136;136m│\033[0m  \033[1m/clear\033[0m                Clear conversation history and screen
\033[38;2;136;136;136m│\033[0m  \033[1m/model [name]\033[0m         Switch active model (flash-next, 27b)
\033[38;2;136;136;136m│\033[0m  \033[1m/think [level]\033[0m        Set reasoning effort (off, low, medium, xhigh)
\033[38;2;136;136;136m│\033[0m  \033[1m/instructions [text]\033[0m  Set or view system instructions
\033[38;2;136;136;136m│\033[0m  \033[1m/save [name]\033[0m          Save session transcript to disk
\033[38;2;136;136;136m│\033[0m  \033[1m/sessions\033[0m             List saved chat sessions
\033[38;2;136;136;136m│\033[0m  \033[1m/resume <name>\033[0m        Resume a saved session
\033[38;2;136;136;136m│\033[0m  \033[1m/info\033[0m                 Display hardware, context, and endpoint info
\033[38;2;136;136;136m╰{dash}╯\033[0m"""
    print(card + "\n")


def print_sessions():
    """Lists saved sessions in an AFM-styled bordered box."""
    sdir = get_sessions_dir()
    files = sorted(sdir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    width = get_term_width()
    dash = "─" * max(2, width - 2)

    if not files:
        print(f"\033[38;2;153;153;153mNo saved sessions found in {sdir}.\033[0m\n")
        return

    header_dash = "─" * max(2, width - 13)
    print(f"\033[38;2;136;136;136m╭─ Sessions {header_dash}╮\033[0m")
    for f in files[:10]:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            s_id = data.get("session_id", f.stem)[:24]
            m_id = data.get("model", "unknown")
            msg_count = len(data.get("messages", []))
            age = time_ago(data.get("saved_at", f.stat().st_mtime))
            row = f"  \033[1m{s_id:<24}\033[0m  {m_id:<12}  {msg_count:>3} msgs  ({age})"
            print(f"\033[38;2;136;136;136m│\033[0m{row}")
        except Exception:
            continue
    print(f"\033[38;2;136;136;136m╰{dash}╯\033[0m\n")


def print_info(session: ChatSession, ctx: int, host: str, port: int, lru: bool):
    """Prints runtime hardware and configuration info."""
    width = get_term_width()
    dash = "─" * max(2, width - 2)
    header_dash = "─" * max(2, width - 9)
    silicon = (
        "Apple Neural Engine (100% pure on-chip)"
        if normalize_model_name(session.model) == "27b"
        else "Apple Neural Engine (GDN/QSA) + MLX (MoE)"
    )
    cache = "Enabled (LRU prefix reuse)" if lru else "Disabled"
    print(f"""\033[38;2;136;136;136m╭─ Info {header_dash}╮\033[0m
\033[38;2;136;136;136m│\033[0m  Model:       \033[1m{session.model_id}\033[0m
\033[38;2;136;136;136m│\033[0m  Hardware:    {silicon}
\033[38;2;136;136;136m│\033[0m  Context:     {format_tokens(ctx)} tokens
\033[38;2;136;136;136m│\033[0m  Endpoint:    http://{host}:{port}/v1
\033[38;2;136;136;136m│\033[0m  Cache:       {cache}
\033[38;2;136;136;136m│\033[0m  Thinking:    {session.thinking}
\033[38;2;136;136;136m│\033[0m  Session:     {session.session_id}
\033[38;2;136;136;136m╰{dash}╯\033[0m\n""")


class AFMPromptReader:
    """Exact reverse-engineered AFM interactive input prompt with live terminal resizing."""

    def __init__(self, model_tag: str = "27b (Qwen 3.8 27B Pure ANE)"):
        self.model_tag = model_tag
        self.history: list[str] = []
        self.resized = False
        self._load_history()
        self._setup_signals()

    def _load_history(self):
        try:
            hist_file = get_qwen_ane_dir() / "history"
            if hist_file.is_file():
                with hist_file.open("r", encoding="utf-8", errors="ignore") as f:
                    self.history = [line.strip() for line in f if line.strip()][-200:]
        except Exception:
            pass

    def _save_history(self):
        try:
            hist_file = get_qwen_ane_dir() / "history"
            hist_file.parent.mkdir(parents=True, exist_ok=True)
            with hist_file.open("w", encoding="utf-8") as f:
                for line in self.history[-500:]:
                    f.write(line + "\n")
        except Exception:
            pass

    def _setup_signals(self):
        def _handle_winch(signum, frame):
            self.resized = True
        try:
            signal.signal(signal.SIGWINCH, _handle_winch)
        except Exception:
            pass

    def read_prompt(self) -> str:
        """Reads user input using raw terminal mode with exact AFM box styling and live resize."""
        if not sys.stdin.isatty() or termios is None or tty is None:
            try:
                line = sys.stdin.readline()
                if not line:
                    raise EOFError
                return line.rstrip("\r\n")
            except (KeyboardInterrupt, EOFError):
                raise

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        buffer: list[str] = []
        cursor = 0
        history_temp = list(self.history)
        hist_idx = len(history_temp)
        saved_draft = ""

        prev_rendered = False
        prev_lines_count = 0

        def get_width() -> int:
            return max(24, shutil.get_terminal_size(fallback=(80, 24)).columns)

        def render():
            nonlocal prev_rendered, prev_lines_count

            width = get_width()
            inner_width = max(10, width - 2)

            text_str = "".join(buffer)
            matches = []
            if text_str.startswith("/"):
                query = text_str.strip().lower()
                if " " not in text_str:
                    matches = [(cmd, desc) for cmd, desc in SLASH_COMMAND_INFO if cmd.lower().startswith(query) and cmd.lower() != query]

            top_border = f"\033[38;2;136;136;136m╭{'─' * inner_width}╮\033[0m\r\n"

            prefix = " > "
            prefix_len = len(prefix)
            max_visible_len = inner_width - prefix_len

            if cursor < max_visible_len:
                view_start = 0
            else:
                view_start = cursor - max_visible_len + 1

            visible_text = text_str[view_start : view_start + max_visible_len]
            pad_len = max(0, max_visible_len - len(visible_text))
            input_row = (
                f"\033[38;2;136;136;136m│\033[0m"
                f"{prefix}{visible_text}{' ' * pad_len}"
                f"\033[38;2;136;136;136m│\033[0m\r\n"
            )

            bottom_border = f"\033[38;2;136;136;136m╰{'─' * inner_width}╯\033[0m\r\n"
            tag_line = f"\033[38;2;153;153;153m {self.model_tag}\033[0m\r\n"

            lines = [top_border, input_row, bottom_border, tag_line]

            if matches:
                for cmd, desc in matches[:6]:
                    lines.append(f"\033[38;2;153;153;153m  \033[1m{cmd:<14}\033[0m\033[38;2;153;153;153m·  {desc}\033[0m\r\n")
                if len(matches) == 1:
                    lines.append(f"\033[38;2;120;120;120m  tab to complete\033[0m\r\n")
                else:
                    lines.append(f"\033[38;2;120;120;120m  tab to complete · ↑↓ to select\033[0m\r\n")

            if prev_rendered:
                # Move cursor from Line 1 (input_row) up to Line 0 (top_border)
                sys.stdout.write("\033[1A\r")

            for l in lines:
                sys.stdout.write("\033[2K" + l)

            if prev_lines_count > len(lines):
                extra = prev_lines_count - len(lines)
                for _ in range(extra):
                    sys.stdout.write("\033[2K\r\n")
                sys.stdout.write(f"\033[{extra}A")

            # Cursor is at line len(lines). Move up to Line 1 (input_row):
            lines_up = len(lines) - 1
            cursor_col = 1 + 1 + prefix_len + (cursor - view_start)
            sys.stdout.write(f"\033[{lines_up}A\033[{cursor_col}G")
            sys.stdout.flush()

            prev_rendered = True
            prev_lines_count = len(lines)

        try:
            tty.setcbreak(fd)
            sys.stdout.write("\033[?25l")
            render()
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()

            while True:
                if self.resized:
                    self.resized = False
                    sys.stdout.write("\033[?25l")
                    render()
                    sys.stdout.write("\033[?25h")
                    sys.stdout.flush()

                try:
                    ch = os.read(fd, 1)
                except InterruptedError:
                    continue

                if not ch:
                    raise EOFError

                if ch in (b"\r", b"\n"):
                    # Enter pressed: finalize box cleanly
                    width = get_width()
                    inner_width = max(10, width - 2)
                    top_border = f"\033[38;2;136;136;136m╭{'─' * inner_width}╮\033[0m\r\n"
                    prefix = " > "
                    max_visible_len = inner_width - len(prefix)
                    text_str = "".join(buffer)
                    vis = text_str[:max_visible_len]
                    pad = max(0, max_visible_len - len(vis))
                    input_row = f"\033[38;2;136;136;136m│\033[0m{prefix}{vis}{' ' * pad}\033[38;2;136;136;136m│\033[0m\r\n"
                    bottom_border = f"\033[38;2;136;136;136m╰{'─' * inner_width}╯\033[0m\r\n"
                    tag_line = f"\033[38;2;153;153;153m {self.model_tag}\033[0m\r\n"

                    if prev_rendered:
                        sys.stdout.write("\033[1A\r")

                    sys.stdout.write("\033[2K" + top_border)
                    sys.stdout.write("\033[2K" + input_row)
                    sys.stdout.write("\033[2K" + bottom_border)
                    sys.stdout.write("\033[2K" + tag_line)

                    extra = prev_lines_count - 4
                    if extra > 0:
                        for _ in range(extra):
                            sys.stdout.write("\033[2K\r\n")
                        sys.stdout.write(f"\033[{extra}A")

                    sys.stdout.write("\r\n")
                    sys.stdout.flush()

                    res = text_str.strip()
                    if res:
                        self.history.append("".join(buffer))
                        self._save_history()
                    return res

                elif ch == b"\x03":  # Ctrl+C
                    lines_below = max(0, prev_lines_count - 2)
                    if lines_below > 0:
                        sys.stdout.write(f"\033[{lines_below}B")
                    sys.stdout.write("\r\n")
                    sys.stdout.flush()
                    raise KeyboardInterrupt

                elif ch == b"\x04":  # Ctrl+D
                    if not buffer:
                        lines_below = max(0, prev_lines_count - 2)
                        if lines_below > 0:
                            sys.stdout.write(f"\033[{lines_below}B")
                        sys.stdout.write("\r\n")
                        sys.stdout.flush()
                        raise EOFError
                    else:
                        if cursor < len(buffer):
                            del buffer[cursor]

                elif ch in (b"\x7f", b"\x08"):  # Backspace
                    if cursor > 0:
                        cursor -= 1
                        del buffer[cursor]

                elif ch == b"\x01":  # Ctrl+A (Home)
                    cursor = 0

                elif ch == b"\x05":  # Ctrl+E (End)
                    cursor = len(buffer)

                elif ch == b"\x15":  # Ctrl+U (Clear before cursor)
                    buffer = buffer[cursor:]
                    cursor = 0

                elif ch == b"\x0b":  # Ctrl+K (Kill to end)
                    buffer = buffer[:cursor]

                elif ch == b"\t":  # Tab completion
                    text_str = "".join(buffer)
                    if text_str.startswith("/"):
                        query = text_str.strip().lower()
                        matches = [cmd for cmd, _ in SLASH_COMMAND_INFO if cmd.lower().startswith(query)]
                        if matches:
                            buffer = list(matches[0] + " ")
                            cursor = len(buffer)

                elif ch == b"\x1b":  # Escape sequence
                    seq = os.read(fd, 2)
                    if seq == b"[A":  # Up arrow
                        if hist_idx > 0:
                            if hist_idx == len(history_temp):
                                saved_draft = "".join(buffer)
                            hist_idx -= 1
                            buffer = list(history_temp[hist_idx])
                            cursor = len(buffer)
                    elif seq == b"[B":  # Down arrow
                        if hist_idx < len(history_temp):
                            hist_idx += 1
                            if hist_idx == len(history_temp):
                                buffer = list(saved_draft)
                            else:
                                buffer = list(history_temp[hist_idx])
                            cursor = len(buffer)
                    elif seq == b"[C":  # Right arrow
                        if cursor < len(buffer):
                            cursor += 1
                    elif seq == b"[D":  # Left arrow
                        if cursor > 0:
                            cursor -= 1
                    elif seq == b"[H":  # Home
                        cursor = 0
                    elif seq == b"[F":  # End
                        cursor = len(buffer)
                    elif seq == b"[3":  # Delete
                        seq2 = os.read(fd, 1)
                        if seq2 == b"~" and cursor < len(buffer):
                            del buffer[cursor]

                else:
                    try:
                        decoded = ch.decode("utf-8")
                        buffer.insert(cursor, decoded)
                        cursor += 1
                    except UnicodeDecodeError:
                        pass

                sys.stdout.write("\033[?25l")
                render()
                sys.stdout.write("\033[?25h")
                sys.stdout.flush()

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def start_chat(
    model: str = "flash-next",
    ctx: int | None = None,
    port: int | None = None,
    host: str = "127.0.0.1",
    lru: bool = True,
    thinking: str = "off",
    system_prompt: str | None = None,
    resume_id: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    model_path: str | Path | None = None,
    hf_repo: str | None = None,
) -> int:
    """Launch the interactive chat REPL with AFM styling."""
    canon = normalize_model_name(model)
    model_id = "Qwen3.8-Flash-Next" if canon == "flash-next" else "Qwen3.8-27B"
    cfg = load_config()

    if port is None:
        port = 1240 if canon == "27b" else cfg.get("default_port", 2457)
    if ctx is None:
        ctx = 4096 if canon == "27b" else cfg.get("default_ctx", 131072)
    elif canon == "27b" and ctx > 4096:
        print(f"\033[38;2;153;153;153mℹ️  Qwen3.8-27B pure ANE engine operates with 4096 token context; setting ctx=4096.\033[0m")
        ctx = 4096

    # 1. Start background inference server if not running
    server_proc = None
    server_log = None
    active = is_server_running(host, port)
    if active:
        running_models = [m.get("id") for m in active.get("data", [])]
        if running_models and not any(canon in rm.lower() or model_id.lower() in rm.lower() for rm in running_models):
            print(f"\033[33m⚠️  A server is already running on http://{host}:{port}/v1 serving {running_models}, but you requested '{canon}'.\033[0m")
            print(f"   Please specify a different port (e.g. --port 1240 or --port 2457) or stop the existing server.")
            return 1
    else:
        print(f"\033[38;2;153;153;153mLoading {model_id} into Apple Neural Engine...\033[0m", flush=True)
        resolved_path = ensure_model(canon, custom_path=model_path, hf_repo=hf_repo)

        env = dict(os.environ)
        env["QWEN_ANE_QUIET"] = "1"
        root = Path(__file__).resolve().parents[1]
        log_path = get_qwen_ane_dir() / "server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        server_log = open(log_path, "a", encoding="utf-8")

        if canon == "flash-next":
            env.update({
                "FLASHNEXT_SPEC": "4",
                "FLASHNEXT_MOE": "mlxresident",
                "FLASHNEXT_HEAD": "mlx",
                "FLASHNEXT_MIL_GDN": "1",
                "FLASHNEXT_MIL_QSA": "1",
                "FLASHNEXT_PREFILL_MIL_K": "32" if lru else "0",
                "FLASHNEXT_MODEL": str(resolved_path),
            })
            cmd = [
                sys.executable,
                "-u",
                str(root / "tools" / "flashnext_server.py"),
                "--host",
                str(host),
                "--port",
                str(port),
                "--ctx",
                str(ctx),
                "--max-new",
                str(max_tokens),
                "--quiet",
            ]
        else:
            env.update({
                "Q38_ANE_REUSE_COMPILED": "1",
                "Q38_ANE_FUSED_TAIL": "1",
                "Q38_ANE_CHAIN_NEXT": "1",
                "Q38_ANE_FUSE_GATE": "0",
                "Q38_ANE_HOST_PREPARE": "1",
                "Q38_ANE_BATCH_ATTN": "16",
                "Q38_ANE_GDN_PREFILL": os.environ.get("Q38_ANE_GDN_PREFILL", "chunk"),
                "Q38_MODEL": str(resolved_path),
            })
            env.pop("PYTHONPATH", None)
            cmd = [
                sys.executable,
                "-u",
                "-P",
                str(root / "tools" / "pure_ane_server.py"),
                "serve",
                "--model",
                str(resolved_path),
                "--host",
                str(host),
                "--port",
                str(port),
                "--context",
                str(ctx),
                "--bits",
                "4",
                "--mtp-draft",
                os.environ.get("Q38_ANE_MTP_DRAFT", "0"),
                "--quiet",
            ]

        server_proc = subprocess.Popen(
            cmd,
            env=env,
            cwd=str(root),
            stdout=server_log,
            stderr=server_log,
        )

        # Wait for server to become ready
        for _ in range(120):
            if is_server_running(host, port):
                break
            if server_proc.poll() is not None:
                print(f"\033[31mError: Server process exited unexpectedly with code {server_proc.returncode}. See {log_path} for details.\033[0m")
                if server_log:
                    server_log.close()
                return 1
            time.sleep(1.0)
        else:
            print(f"\033[31mError: Server timed out while starting. See {log_path} for details.\033[0m")
            if server_proc:
                server_proc.terminate()
            if server_log:
                server_log.close()
            return 1

    session = ChatSession(
        model=canon,
        host=host,
        port=port,
        system_prompt=system_prompt,
        thinking=thinking,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if resume_id:
        if session.load(resume_id):
            print(f"\033[38;2;130;215;90m✓ Resumed session '{resume_id}' ({len(session.messages)} messages)\033[0m")
        else:
            print(f"\033[38;2;220;120;120mSession '{resume_id}' not found. Starting fresh session.\033[0m")

    reader = AFMPromptReader(get_model_badge(canon))
    print_banner(canon)

    last_sigint_time = 0.0

    try:
        while True:
            try:
                raw_input = reader.read_prompt()
            except KeyboardInterrupt:
                now = time.time()
                if now - last_sigint_time < 2.5:
                    print("\n\033[38;2;153;153;153mExiting.\033[0m")
                    break
                last_sigint_time = now
                print("\033[38;2;153;153;153m(Press Ctrl+C again or /exit to quit)\033[0m\n")
                continue
            except EOFError:
                print("\n\033[38;2;153;153;153mExiting.\033[0m")
                break

            user_input = raw_input.strip()
            if not user_input:
                continue

            # Multi-line continuation with backslash \
            while user_input.endswith("\\"):
                user_input = user_input[:-1].strip()
                try:
                    cont = reader.read_prompt()
                    user_input += "\n" + cont
                except (KeyboardInterrupt, EOFError):
                    break

            # Slash commands
            if user_input in ("/exit", "/quit"):
                break

            elif user_input == "/clear":
                session.messages = []
                print("\033[2J\033[H", end="")
                print_banner(session.model)
                continue

            elif user_input == "/help":
                print_help()
                continue

            elif user_input == "/sessions":
                print_sessions()
                continue

            elif user_input == "/info":
                print_info(session, ctx, host, port, lru)
                continue

            elif user_input.startswith("/save"):
                parts = user_input.split(maxsplit=1)
                save_name = parts[1] if len(parts) > 1 else None
                saved_path = session.save(save_name)
                print(f"\033[38;2;130;215;90m✓ Session saved as '{saved_path.stem}'\033[0m\n")
                continue

            elif user_input.startswith("/resume"):
                parts = user_input.split(maxsplit=1)
                if len(parts) < 2:
                    print("\033[38;2;220;120;120mUsage: /resume <session-id-or-name>\033[0m\n")
                else:
                    if session.load(parts[1]):
                        print(f"\033[38;2;130;215;90m✓ Resumed session '{parts[1]}' ({len(session.messages)} messages)\033[0m\n")
                    else:
                        print(f"\033[38;2;220;120;120mSession '{parts[1]}' not found.\033[0m\n")
                continue

            elif user_input.startswith("/think"):
                parts = user_input.split()
                if len(parts) == 1:
                    print(f"\033[38;2;153;153;153mThinking level: {session.thinking}\033[0m\n")
                elif parts[1] in ("off", "low", "medium", "xhigh"):
                    session.thinking = parts[1]
                    print(f"\033[38;2;130;215;90m✓ Thinking set to: {session.thinking}\033[0m\n")
                else:
                    print("\033[38;2;220;120;120mUsage: /think [off|low|medium|xhigh]\033[0m\n")
                continue

            elif user_input.startswith("/instructions"):
                parts = user_input.split(maxsplit=1)
                if len(parts) == 1:
                    current = next((m["content"] for m in session.messages if m.get("role") == "system"), "(none)")
                    print(f"\033[38;2;153;153;153mInstructions: {current}\033[0m\n")
                else:
                    inst_text = parts[1]
                    session.messages = [m for m in session.messages if m.get("role") != "system"]
                    session.messages.insert(0, {"role": "system", "content": inst_text})
                    print(f"\033[38;2;130;215;90m✓ Instructions updated.\033[0m\n")
                continue

            elif user_input.startswith("/model"):
                parts = user_input.split()
                if len(parts) == 1:
                    print(f"\033[38;2;153;153;153mActive model: {session.model_id} ({silicon}) on port {session.port}\033[0m\n")
                else:
                    target_model = normalize_model_name(parts[1])
                    if target_model in ("flash-next", "27b"):
                        session.model = target_model
                        session.model_id = "Qwen3.8-Flash-Next" if target_model == "flash-next" else "Qwen3.8-27B"
                        session.port = 1240 if target_model == "27b" else 2457
                        silicon = "pure ANE" if target_model == "27b" else "ANE + GPU"
                        reader.model_tag = get_model_badge(target_model)
                        print(f"\033[38;2;130;215;90m✓ Switched model to: {session.model_id} (port {session.port})\033[0m\n")
                    else:
                        print("\033[38;2;220;120;120mUnknown model. Choose 'flash-next' or '27b'.\033[0m\n")
                continue

            elif user_input.startswith("/"):
                print(f"\033[38;2;220;120;120mUnknown command '{user_input}'. Type /help for available commands.\033[0m\n")
                continue

            # Model Turn Generation
            session.messages.append({"role": "user", "content": user_input})
            t0 = time.perf_counter()
            first_token_time = None
            token_count = 0
            full_reply = []
            full_thought = []
            in_thinking = False

            renderer = StreamingMarkdownRenderer()

            try:
                for delta in stream_chat_completion(
                    host=session.host,
                    port=session.port,
                    model_id=session.model_id,
                    messages=session.messages,
                    thinking=session.thinking,
                    temperature=session.temperature,
                    max_tokens=session.max_tokens,
                ):
                    if first_token_time is None:
                        first_token_time = time.perf_counter()

                    reasoning = delta.get("reasoning_content")
                    if reasoning:
                        if not in_thinking:
                            print("\033[38;2;150;150;240m⠋ Thinking...\033[0m\n\033[38;2;140;140;150m\033[3m", end="", flush=True)
                            in_thinking = True
                        print(reasoning, end="", flush=True)
                        full_thought.append(reasoning)

                    content = delta.get("content")
                    if content:
                        if in_thinking:
                            print("\033[0m\n\n", end="", flush=True)
                            in_thinking = False
                        renderer.write(content)
                        full_reply.append(content)
                        token_count += 1

                if in_thinking:
                    print("\033[0m\n", flush=True)
                renderer.finish()

                elapsed = max(0.001, time.perf_counter() - t0)
                ttft_ms = int((first_token_time - t0) * 1000) if first_token_time else 0
                tok_per_sec = token_count / elapsed

                reply_text = "".join(full_reply)
                msg: dict[str, Any] = {"role": "assistant", "content": reply_text}
                if full_thought:
                    msg["reasoning_content"] = "".join(full_thought)
                session.messages.append(msg)
                session.save()

                # AFM-style subtle turn metrics badge
                print(f"\033[38;2;120;120;120m {token_count} tokens · {tok_per_sec:.1f} tok/s · TTFT {ttft_ms}ms · {elapsed:.2f}s\033[0m\n")

            except KeyboardInterrupt:
                print(f"\n\033[38;2;220;120;120m [Cancelled.]\033[0m\n")
                if full_reply:
                    session.messages.append({"role": "assistant", "content": "".join(full_reply)})
                    session.save()
            except Exception as exc:
                print(f"\n\033[31mError during completion: {exc}\033[0m\n")

    finally:
        session.save()
        print(f"\033[38;2;153;153;153mSession saved as {session.session_id}.\033[0m")
        print(f"\033[38;2;153;153;153mResume later with: qwen-ane chat --resume {session.session_id}\033[0m\n")
        if server_proc:
            print("\033[38;2;153;153;153mStopping background server...\033[0m")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        if server_log:
            try:
                server_log.close()
            except Exception:
                pass

    return 0
