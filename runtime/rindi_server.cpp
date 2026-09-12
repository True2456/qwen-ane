/*
 * SPDX-License-Identifier: Apache-2.0
 * runtime/rindi_server.cpp - Pure C++ Standalone Native Inference Server with Real-Time TUI Dashboard.
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <atomic>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <csignal>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cerrno>

#include "rindi_tui.h"
#include "rindi_engine.h"
#include "rindi_native_chain.h"
#include "metal_engine.h"

static std::atomic<bool> g_running{true};
static RindiTUI* g_tui = nullptr;
static RindiEngine* g_engine = nullptr;
static std::mutex g_engine_mutex;

static std::string trim_copy(const std::string& input) {
    const auto first = std::find_if_not(input.begin(), input.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    const auto last = std::find_if_not(input.rbegin(), input.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    return last <= first ? std::string() : std::string(first, last);
}

static int run_local_chat(RindiEngine* engine, const ServerConfig& config) {
    if (!engine || !engine->is_ready()) return 1;

    const bool interactive_input = isatty(STDIN_FILENO);
    if (interactive_input) {
        // Do not assume a previous dashboard/process restored every terminal
        // bit. In particular, ICANON with ICRNL disabled makes Return produce
        // a literal CR that never completes std::getline(). Establish a full
        // cooked line discipline for focused chat and leave the shell sane on
        // exit as well.
        struct termios cooked{};
        if (tcgetattr(STDIN_FILENO, &cooked) == 0) {
            cooked.c_iflag |= (ICRNL | BRKINT);
            cooked.c_iflag &= ~(IGNCR | INLCR);
            cooked.c_lflag |= (ICANON | ECHO | ISIG | IEXTEN);
            cooked.c_oflag |= (OPOST | ONLCR);
            cooked.c_cc[VEOF] = 4;
            cooked.c_cc[VEOL] = 0;
            cooked.c_cc[VMIN] = 1;
            cooked.c_cc[VTIME] = 0;
            tcsetattr(STDIN_FILENO, TCSANOW, &cooked);
        }
    }

    const bool color = isatty(STDOUT_FILENO);
    const char* green = color ? "\033[92m" : "";
    const char* white = color ? "\033[97m" : "";
    const char* dim = color ? "\033[2m" : "";
    const char* bold = color ? "\033[1m" : "";
    const char* reset = color ? "\033[0m" : "";
    if (color) std::cout << "\033[2J\033[H";
    std::cout << bold << green << "Rindi Chat" << reset << "  "
              << dim << config.model_name << " · ANE prefill · Metal decode" << reset << "\n"
              << dim << "Multi-turn APC enabled. Type /help for commands." << reset << "\n\n";

    std::vector<std::pair<std::string, std::string>> history;
    int max_tokens = 512;
    if (const char* value = std::getenv("RINDI_CHAT_MAX_TOKENS")) {
        max_tokens = std::clamp(std::atoi(value), 1, 8192);
    }
    float temperature = 0.7f;

    while (g_running.load()) {
        std::cout << bold << white << "You › " << reset << std::flush;
        std::string input;
        if (!std::getline(std::cin, input)) break;
        input = trim_copy(input);
        if (input.empty()) continue;

        if (input == "/quit" || input == "/exit" || input == "quit" || input == "exit") {
            break;
        }
        if (input == "/clear") {
            history.clear();
            std::cout << dim << "Conversation cleared." << reset << "\n\n";
            continue;
        }
        if (input == "/stats") {
            const GenerationStats& stats = engine->get_last_stats();
            std::cout << dim << "prompt=" << stats.prompt_tokens
                      << " prefill=" << stats.prefill_tps() << " tok/s"
                      << " decode=" << stats.decode_tps() << " tok/s"
                      << " APC=" << (engine->last_apc_hit() ? "hit" : "miss")
                      << " reused=" << engine->last_apc_saved() << reset << "\n\n";
            continue;
        }
        if (input.rfind("/tokens ", 0) == 0) {
            max_tokens = std::clamp(std::atoi(input.c_str() + 8), 1, 8192);
            std::cout << dim << "Maximum response tokens: " << max_tokens << reset << "\n\n";
            continue;
        }
        if (input.rfind("/temp ", 0) == 0) {
            temperature = std::clamp(std::strtof(input.c_str() + 6, nullptr), 0.0f, 2.0f);
            std::cout << dim << "Temperature: " << temperature << reset << "\n\n";
            continue;
        }
        if (input == "/help") {
            std::cout << dim
                      << "/clear      start a fresh conversation\n"
                      << "/stats      show last-turn performance and cache status\n"
                      << "/tokens N   set maximum response length\n"
                      << "/temp N     set sampling temperature\n"
                      << "/quit       leave chat"
                      << reset << "\n\n";
            continue;
        }
        if (input.rfind("/chat ", 0) == 0) {
            input = trim_copy(input.substr(6));
            if (input.empty()) continue;
        }

        history.push_back({"user", input});
        std::cout << bold << green << "Rindi › " << reset << std::flush;
        std::string response = engine->chat_completion(
            history, "", max_tokens, temperature,
            [](const std::string& token) {
                std::cout << token << std::flush;
            }, false);
        std::cout << "\n";
        if (response.empty()) {
            history.pop_back();
            std::cout << dim << "Generation failed; the turn was not added to history."
                      << reset << "\n\n";
            continue;
        }
        history.push_back({"assistant", response});

        const GenerationStats& stats = engine->get_last_stats();
        std::cout << dim << "[" << stats.prompt_tokens << " prompt · "
                  << stats.prefill_tps() << " prefill tok/s · "
                  << stats.decode_tps() << " decode tok/s";
        if (engine->last_apc_hit()) {
            std::cout << " · APC reused " << engine->last_apc_saved();
        }
        std::cout << "]" << reset << "\n\n";
    }

    std::cout << dim << "Chat closed." << reset << "\n";
    return 0;
}

void signal_handler(int signum) {
    if (g_tui) {
        g_tui->log("Received termination signal (" + std::to_string(signum) + "), shutting down...", "INFO");
        g_tui->stop();
    }
    g_running.store(false);
}

// ---------------------------------------------------------------------------
// Lightweight, Robust JSON Utilities
// ---------------------------------------------------------------------------

static std::string json_escape(const std::string& input) {
    std::string out;
    out.reserve(input.size() + 16);
    for (char c : input) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
                break;
        }
    }
    return out;
}

static std::string json_unescape(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        if (input[i] == '\\' && i + 1 < input.size()) {
            char next = input[i + 1];
            if (next == '"') { out += '"'; i++; }
            else if (next == '\\') { out += '\\'; i++; }
            else if (next == '/') { out += '/'; i++; }
            else if (next == 'b') { out += '\b'; i++; }
            else if (next == 'f') { out += '\f'; i++; }
            else if (next == 'n') { out += '\n'; i++; }
            else if (next == 'r') { out += '\r'; i++; }
            else if (next == 't') { out += '\t'; i++; }
            else if (next == 'u' && i + 5 < input.size()) {
                std::string hex_str = input.substr(i + 2, 4);
                try {
                    unsigned int cp = std::stoul(hex_str, nullptr, 16);
                    if (cp < 0x80) {
                        out += static_cast<char>(cp);
                    } else if (cp < 0x800) {
                        out += static_cast<char>(0xC0 | ((cp >> 6) & 0x1F));
                        out += static_cast<char>(0x80 | (cp & 0x3F));
                    } else {
                        out += static_cast<char>(0xE0 | ((cp >> 12) & 0x0F));
                        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
                        out += static_cast<char>(0x80 | (cp & 0x3F));
                    }
                } catch (...) {
                    out += "?";
                }
                i += 5;
            } else {
                out += next;
                i++;
            }
        } else {
            out += input[i];
        }
    }
    return out;
}

struct ChatMessage {
    std::string role;
    std::string content;
    std::string tool_call_id;
    std::string name;
};

struct ToolDef {
    std::string name;
    std::string description;
    std::string parameters_schema;
};

static std::string extract_json_field(const std::string& json, const std::string& key) {
    std::string search_key = "\"" + key + "\"";
    size_t key_pos = json.find(search_key);
    if (key_pos == std::string::npos) return "";

    size_t colon_pos = json.find(':', key_pos + search_key.size());
    if (colon_pos == std::string::npos) return "";

    size_t val_start = json.find_first_not_of(" \t\r\n", colon_pos + 1);
    if (val_start == std::string::npos) return "";

    if (json[val_start] == '"') {
        size_t cur = val_start + 1;
        while (cur < json.size()) {
            if (json[cur] == '\\') {
                cur += 2;
                continue;
            }
            if (json[cur] == '"') {
                return json_unescape(json.substr(val_start + 1, cur - (val_start + 1)));
            }
            cur++;
        }
        return "";
    } else if (json[val_start] == '{' || json[val_start] == '[') {
        char open_char = json[val_start];
        char close_char = (open_char == '{') ? '}' : ']';
        int depth = 0;
        bool in_str = false;
        for (size_t cur = val_start; cur < json.size(); ++cur) {
            if (json[cur] == '"' && (cur == 0 || json[cur - 1] != '\\')) {
                in_str = !in_str;
            }
            if (!in_str) {
                if (json[cur] == open_char) depth++;
                else if (json[cur] == close_char) {
                    depth--;
                    if (depth == 0) {
                        return json.substr(val_start, cur - val_start + 1);
                    }
                }
            }
        }
        return "";
    } else {
        size_t val_end = json.find_first_of(",}\n\r \t", val_start);
        if (val_end == std::string::npos) val_end = json.size();
        return json.substr(val_start, val_end - val_start);
    }
}

static std::string normalize_message_content(const std::string& raw) {
    if (raw.empty()) return "";
    if (raw.front() != '[') return raw;
    std::string result;
    size_t pos = 0;
    while (pos < raw.size()) {
        size_t obj_start = raw.find('{', pos);
        if (obj_start == std::string::npos) break;
        size_t obj_end = raw.find('}', obj_start);
        if (obj_end == std::string::npos) break;
        std::string elem = raw.substr(obj_start, obj_end - obj_start + 1);
        std::string text = extract_json_field(elem, "text");
        if (!text.empty()) {
            if (!result.empty()) result += "\n";
            result += text;
        }
        pos = obj_end + 1;
    }
    return result.empty() ? raw : result;
}

static std::vector<ChatMessage> parse_messages(const std::string& json_body) {
    std::vector<ChatMessage> messages;
    std::string messages_arr = extract_json_field(json_body, "messages");
    if (messages_arr.empty() || messages_arr.front() != '[') return messages;

    size_t i = 1;
    while (i < messages_arr.size()) {
        size_t obj_start = messages_arr.find('{', i);
        if (obj_start == std::string::npos) break;

        int depth = 0;
        bool in_str = false;
        size_t obj_end = std::string::npos;
        for (size_t cur = obj_start; cur < messages_arr.size(); ++cur) {
            if (messages_arr[cur] == '"' && (cur == 0 || messages_arr[cur - 1] != '\\')) {
                in_str = !in_str;
            }
            if (!in_str) {
                if (messages_arr[cur] == '{') depth++;
                else if (messages_arr[cur] == '}') {
                    depth--;
                    if (depth == 0) {
                        obj_end = cur;
                        break;
                    }
                }
            }
        }

        if (obj_end == std::string::npos) break;
        std::string obj_str = messages_arr.substr(obj_start, obj_end - obj_start + 1);

        ChatMessage msg;
        msg.role = extract_json_field(obj_str, "role");
        msg.content = normalize_message_content(extract_json_field(obj_str, "content"));
        msg.tool_call_id = extract_json_field(obj_str, "tool_call_id");
        msg.name = extract_json_field(obj_str, "name");

        if (!msg.role.empty()) {
            messages.push_back(msg);
        }
        i = obj_end + 1;
    }
    return messages;
}

static std::vector<ToolDef> parse_tools(const std::string& json_body) {
    std::vector<ToolDef> tools;
    std::string tools_arr = extract_json_field(json_body, "tools");
    if (tools_arr.empty() || tools_arr.front() != '[') return tools;

    size_t i = 1;
    while (i < tools_arr.size()) {
        size_t obj_start = tools_arr.find('{', i);
        if (obj_start == std::string::npos) break;

        int depth = 0;
        bool in_str = false;
        size_t obj_end = std::string::npos;
        for (size_t cur = obj_start; cur < tools_arr.size(); ++cur) {
            if (tools_arr[cur] == '"' && (cur == 0 || tools_arr[cur - 1] != '\\')) {
                in_str = !in_str;
            }
            if (!in_str) {
                if (tools_arr[cur] == '{') depth++;
                else if (tools_arr[cur] == '}') {
                    depth--;
                    if (depth == 0) {
                        obj_end = cur;
                        break;
                    }
                }
            }
        }

        if (obj_end == std::string::npos) break;
        std::string obj_str = tools_arr.substr(obj_start, obj_end - obj_start + 1);

        std::string fn_obj = extract_json_field(obj_str, "function");
        if (!fn_obj.empty()) {
            ToolDef td;
            td.name = extract_json_field(fn_obj, "name");
            td.description = extract_json_field(fn_obj, "description");
            td.parameters_schema = extract_json_field(fn_obj, "parameters");
            if (!td.name.empty()) {
                tools.push_back(td);
            }
        }
        i = obj_end + 1;
    }
    return tools;
}

static std::string build_tools_json(const std::vector<ToolDef>& tools) {
    if (tools.empty()) return "";
    std::string out = "[";
    for (size_t i = 0; i < tools.size(); ++i) {
        if (i) out += ",";
        out += "{\"type\":\"function\",\"function\":{\"name\":\"" +
               json_escape(tools[i].name) + "\",\"description\":\"" +
               json_escape(tools[i].description) + "\",\"parameters\":" +
               (tools[i].parameters_schema.empty() ? "{}" : tools[i].parameters_schema) +
               "}}";
    }
    out += "]";
    return out;
}

struct ParsedToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;  // compact JSON string of arguments
};

// Qwen emits zero or more <tool_call>{"name":..., "arguments":{...}}</tool_call>
// blocks in its content. Extract them into OpenAI-compatible entries and
// return the remaining content. Unparsable blocks stay in the content.
static void parse_qwen_tool_calls(const std::string& text,
                                  std::string& content_out,
                                  std::vector<ParsedToolCall>& calls_out) {
    static std::atomic<uint64_t> call_counter{0};
    content_out.clear();
    calls_out.clear();
    size_t pos = 0;
    while (true) {
        const size_t start = text.find("<tool_call>", pos);
        if (start == std::string::npos) break;
        const size_t end = text.find("</tool_call>", start);
        const size_t block_end = (end == std::string::npos) ? text.size() : end;
        std::string body = text.substr(start + 11, block_end - (start + 11));
        // Trim whitespace/newlines around the JSON payload.
        const size_t b0 = body.find_first_not_of(" \t\r\n");
        const size_t b1 = body.find_last_not_of(" \t\r\n");
        bool parsed = false;
        if (b0 != std::string::npos && b1 != std::string::npos && b1 >= b0) {
            body = body.substr(b0, b1 - b0 + 1);
            if (!body.empty() && body.front() == '{') {
                const std::string name = extract_json_field(body, "name");
                std::string args = extract_json_field(body, "arguments");
                if (!name.empty()) {
                    ParsedToolCall tc;
                    tc.id = "call_" + std::to_string(++call_counter);
                    tc.name = name;
                    tc.arguments_json = args.empty() ? "{}" : args;
                    calls_out.push_back(std::move(tc));
                    parsed = true;
                }
            }
            // XML format: <function=example_fn>\n<parameter=param_name>\nval\n</parameter>\n</function>
            size_t fn_tag = body.find("<function=");
            if (!parsed && fn_tag != std::string::npos) {
                size_t fn_start = fn_tag + 10;
                size_t fn_end = body.find('>', fn_start);
                if (fn_end != std::string::npos) {
                    std::string fn_name = body.substr(fn_start, fn_end - fn_start);
                    while (!fn_name.empty() && (fn_name.back() == '"' || fn_name.back() == ' ' || fn_name.back() == '\n' || fn_name.back() == '\r')) fn_name.pop_back();
                    while (!fn_name.empty() && (fn_name.front() == '"' || fn_name.front() == ' ' || fn_name.front() == '\n' || fn_name.front() == '\r')) fn_name.erase(fn_name.begin());

                    std::string args_json = "{";
                    size_t ppos = fn_end + 1;
                    bool first_arg = true;
                    while (true) {
                        size_t ptag = body.find("<parameter=", ppos);
                        if (ptag == std::string::npos) break;
                        size_t pname_start = ptag + 11;
                        size_t pname_end = body.find('>', pname_start);
                        if (pname_end == std::string::npos) break;
                        std::string pname = body.substr(pname_start, pname_end - pname_start);
                        while (!pname.empty() && (pname.back() == '"' || pname.back() == ' ')) pname.pop_back();
                        while (!pname.empty() && (pname.front() == '"' || pname.front() == ' ')) pname.erase(pname.begin());

                        size_t pend = body.find("</parameter>", pname_end);
                        std::string pval = (pend != std::string::npos)
                            ? body.substr(pname_end + 1, pend - (pname_end + 1))
                            : body.substr(pname_end + 1);

                        while (!pval.empty() && (pval.back() == '\n' || pval.back() == '\r' || pval.back() == ' ' || pval.back() == '\t')) pval.pop_back();
                        while (!pval.empty() && (pval.front() == '\n' || pval.front() == '\r' || pval.front() == ' ' || pval.front() == '\t')) pval.erase(pval.begin());

                        if (!first_arg) args_json += ",";
                        args_json += "\"" + json_escape(pname) + "\":";
                        if (!pval.empty() && (pval.front() == '{' || pval.front() == '[' || pval == "true" || pval == "false" || pval == "null" || (pval.front() >= '0' && pval.front() <= '9'))) {
                            args_json += pval;
                        } else {
                            args_json += "\"" + json_escape(pval) + "\"";
                        }
                        first_arg = false;
                        ppos = (pend != std::string::npos) ? pend + 12 : body.size();
                    }
                    args_json += "}";

                    if (!fn_name.empty()) {
                        ParsedToolCall tc;
                        tc.id = "call_" + std::to_string(++call_counter);
                        tc.name = fn_name;
                        tc.arguments_json = args_json;
                        calls_out.push_back(std::move(tc));
                        parsed = true;
                    }
                }
            }
        }
        if (!parsed) {
            // Keep the raw block in the content rather than dropping model output.
            content_out.append(text, pos, block_end - pos);
        } else {
            content_out.append(text, pos, start - pos);
        }
        pos = (end == std::string::npos) ? text.size() : end + 12;
        if (end == std::string::npos) break;
    }
    content_out.append(text, pos, text.size() - pos);
}

// Routes live model deltas into reasoning (<think>...</think>) and content
// streams based on what the model actually emitted. No synthetic reasoning
// text is ever injected.
class ThinkSplitter {
public:
    using EmitFn = std::function<void(const std::string&)>;
    ThinkSplitter(EmitFn reasoning_cb, EmitFn content_cb, bool enable_thinking = true)
        : reasoning_cb_(std::move(reasoning_cb)), content_cb_(std::move(content_cb)),
          mode_(enable_thinking ? Mode::kReasoning : Mode::kContent) {}

    void feed(const std::string& delta) {
        pending_ += delta;
        if (mode_ == Mode::kReasoning) {
            // In reasoning mode: check for closing tag '</think>'
            static const std::string kClose = "</think>";
            const size_t close = pending_.find(kClose);
            if (close != std::string::npos) {
                emit_reasoning(pending_.substr(0, close));
                pending_ = pending_.substr(close + kClose.size());
                // Trim leading newlines immediately following </think>
                while (!pending_.empty() && (pending_.front() == '\n' || pending_.front() == '\r')) {
                    pending_.erase(pending_.begin());
                }
                mode_ = Mode::kContent;
                if (!pending_.empty()) {
                    emit_clean(pending_, true);
                    pending_.clear();
                }
                return;
            }
            // Safely hold back up to kClose.size() - 1 chars in case a partial '</think' is straddling chunks
            const size_t safe = (pending_.size() >= kClose.size())
                ? pending_.size() - (kClose.size() - 1) : 0;
            if (safe > 0) {
                emit_reasoning(pending_.substr(0, safe));
                pending_ = pending_.substr(safe);
            }
            return;
        }

        // Mode::kContent:
        emit_clean(pending_, true);
        pending_.clear();
    }

    void finish() {
        if (pending_.empty()) return;
        if (mode_ == Mode::kReasoning) emit_reasoning(pending_);
        else emit_clean(pending_, true);
        pending_.clear();
    }

    const std::string& content_accum() const { return content_accum_; }
    const std::string& reasoning_accum() const { return reasoning_accum_; }

private:
    enum class Mode { kReasoning, kContent };

    void emit_reasoning(const std::string& s) {
        if (s.empty()) return;
        std::string clean;
        clean.reserve(s.size());
        for (size_t i = 0; i < s.size();) {
            if (s.compare(i, 7, "<think>") == 0) { i += 7; continue; }
            if (s.compare(i, 8, "</think>") == 0) { i += 8; continue; }
            if (s.compare(i, 10, "<|im_end|>") == 0) { i += 10; continue; }
            if (s.compare(i, 12, "<|im_start|>") == 0) { i += 12; continue; }
            if (s.compare(i, 13, "<|endoftext|>") == 0) { i += 13; continue; }
            clean += s[i++];
        }
        if (clean.empty()) return;
        reasoning_accum_ += clean;
        if (reasoning_cb_) reasoning_cb_(clean);
    }
    void emit_clean(const std::string& s, bool to_stream) {
        if (s.empty()) return;
        // Strip any stray think markers and stop tokens from content.
        std::string clean;
        clean.reserve(s.size());
        for (size_t i = 0; i < s.size();) {
            if (s.compare(i, 7, "<think>") == 0) { i += 7; continue; }
            if (s.compare(i, 8, "</think>") == 0) { i += 8; continue; }
            if (s.compare(i, 10, "<|im_end|>") == 0) { i += 10; continue; }
            if (s.compare(i, 12, "<|im_start|>") == 0) { i += 12; continue; }
            if (s.compare(i, 13, "<|endoftext|>") == 0) { i += 13; continue; }
            clean += s[i++];
        }
        if (clean.empty()) return;
        content_accum_ += clean;
        if (to_stream && content_cb_) content_cb_(clean);
    }

    EmitFn reasoning_cb_;
    EmitFn content_cb_;
    Mode mode_{Mode::kReasoning};
    std::string pending_;
    std::string content_accum_;
    std::string reasoning_accum_;
};

// ---------------------------------------------------------------------------
// HTTP Response Helpers
// ---------------------------------------------------------------------------

static void send_cors_preflight(int client_fd) {
    std::string resp = "HTTP/1.1 204 No Content\r\n"
                       "Access-Control-Allow-Origin: *\r\n"
                       "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                       "Access-Control-Allow-Headers: Authorization, Content-Type, Accept, Origin, User-Agent, X-Requested-With\r\n"
                       "Access-Control-Max-Age: 86400\r\n"
                       "Content-Length: 0\r\n"
                       "Connection: keep-alive\r\n\r\n";
    write(client_fd, resp.c_str(), resp.size());
}

static void send_json_response(int client_fd, int status_code, const std::string& status_text, const std::string& json_body) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status_code << " " << status_text << "\r\n"
        << "Content-Type: application/json\r\n"
        << "Access-Control-Allow-Origin: *\r\n"
        << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        << "Access-Control-Allow-Headers: *\r\n"
        << "Content-Length: " << json_body.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << json_body;
    std::string resp = oss.str();
    write(client_fd, resp.c_str(), resp.size());
}

// ---------------------------------------------------------------------------
// Client Request Handler
// ---------------------------------------------------------------------------

void handle_client(int client_fd, RindiEngine* engine) {
    std::string req;
    char buffer[8192];
    size_t content_length = 0;
    bool headers_done = false;

    struct timeval tv;
    tv.tv_sec = 10;
    tv.tv_usec = 0;
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);

    while (true) {
        ssize_t bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
        if (bytes_read <= 0) break;
        buffer[bytes_read] = '\0';
        req.append(buffer, bytes_read);

        if (!headers_done) {
            size_t header_end = req.find("\r\n\r\n");
            if (header_end != std::string::npos) {
                headers_done = true;
                std::string lower_req = req.substr(0, header_end);
                std::transform(lower_req.begin(), lower_req.end(), lower_req.begin(), ::tolower);
                size_t cl_pos = lower_req.find("content-length:");
                if (cl_pos != std::string::npos) {
                    size_t num_start = lower_req.find_first_of("0123456789", cl_pos);
                    if (num_start != std::string::npos) {
                        size_t num_end = lower_req.find_first_not_of("0123456789", num_start);
                        content_length = std::stoul(lower_req.substr(num_start, num_end - num_start));
                    }
                }
            }
        }

        if (headers_done) {
            size_t header_end = req.find("\r\n\r\n");
            size_t body_len = req.size() - (header_end + 4);
            if (body_len >= content_length) {
                break;
            }
        }
    }

    if (req.empty()) {
        close(client_fd);
        return;
    }

    // 1. Handle OPTIONS (CORS Preflight)
    if (req.rfind("OPTIONS", 0) == 0) {
        send_cors_preflight(client_fd);
        close(client_fd);
        return;
    }

    // 2. Handle GET /v1/models or GET /models
    if (req.find("GET /v1/models") != std::string::npos || req.find("GET /models") != std::string::npos) {
        if (g_tui) g_tui->log("GET /v1/models - returned model list", "HTTP");
        const size_t context = engine ? engine->context_length() : 128 * 1024;
        const std::string suffix = "\",\"object\":\"model\",\"created\":1787300000,"
                                   "\"owned_by\":\"rindi\",\"context_window\":" +
                                   std::to_string(context) +
                                   ",\"max_output_tokens\":8192}";
        std::string body = "{\"object\":\"list\",\"data\":["
                           "{\"id\":\"Qwen3.8-27B" + suffix + ","
                           "{\"id\":\"rindi" + suffix + "]}";
        send_json_response(client_fd, 200, "OK", body);
        close(client_fd);
        return;
    }

    // 3. Handle POST /v1/chat/completions or POST /chat/completions
    if (req.find("POST /v1/chat/completions") != std::string::npos || req.find("POST /chat/completions") != std::string::npos) {
        size_t header_end = req.find("\r\n\r\n");
        std::string body = (header_end != std::string::npos) ? req.substr(header_end + 4) : "";

        bool stream = (body.find("\"stream\": true") != std::string::npos || body.find("\"stream\":true") != std::string::npos);
        std::string model_requested = extract_json_field(body, "model");
        if (model_requested.empty()) model_requested = "Qwen3.8-27B";

        // Honor the OpenAI-compatible generation limit.  The previous code
        // hard-coded 128 below, which made clients' max_tokens setting
        // ineffective and could produce responses longer than requested.
        int requested_max_tokens = 128;
        std::string max_tokens_field = extract_json_field(body, "max_tokens");
        if (max_tokens_field.empty()) {
            max_tokens_field = extract_json_field(body, "max_completion_tokens");
        }
        if (!max_tokens_field.empty()) {
            try {
                requested_max_tokens = std::stoi(max_tokens_field);
            } catch (...) {
                requested_max_tokens = 128;
            }
        }
        requested_max_tokens = std::max(1, std::min(requested_max_tokens, 32768));

        float requested_temperature = 0.7f;
        std::string temperature_field = extract_json_field(body, "temperature");
        if (!temperature_field.empty()) {
            try {
                requested_temperature = std::stof(temperature_field);
            } catch (...) {
                requested_temperature = 0.7f;
            }
        }
        if (!std::isfinite(requested_temperature) || requested_temperature < 0.0f)
            requested_temperature = 0.0f;

        bool enable_thinking = true;
        // Pi sends chat-template controls in a nested object. Accept both
        // that OpenAI-compatible shape and the flat fields used by curl.
        std::string template_kwargs = extract_json_field(body, "chat_template_kwargs");
        std::string thinking_field = extract_json_field(body, "enable_thinking");
        if (thinking_field.empty() && !template_kwargs.empty()) {
            thinking_field = extract_json_field(template_kwargs, "enable_thinking");
        }
        if (thinking_field == "false" || thinking_field == "0")
            enable_thinking = false;
        std::string reasoning_effort = extract_json_field(body, "reasoning_effort");
        if (reasoning_effort.empty() && !template_kwargs.empty()) {
            reasoning_effort = extract_json_field(template_kwargs, "reasoning_effort");
        }
        if (reasoning_effort.empty()) reasoning_effort = "xhigh";
        if (reasoning_effort == "off" || reasoning_effort == "none")
            enable_thinking = false;

        std::vector<ChatMessage> messages = parse_messages(body);
        std::vector<ToolDef> tools = parse_tools(body);

        std::string last_user_prompt;
        std::vector<std::pair<std::string, std::string>> formatted_msgs;

        for (const auto& m : messages) {
            formatted_msgs.push_back({m.role, m.content});
            if (m.role == "user") {
                last_user_prompt = m.content;
            }
        }

        std::string prompt_snippet = last_user_prompt.empty() ? "chat request" : last_user_prompt.substr(0, std::min((size_t)48, last_user_prompt.size()));

        if (g_tui) {
            g_tui->log("POST /v1/chat/completions [stream=" + std::string(stream ? "true" : "false") + "] (\"" + prompt_snippet + "\")", "HTTP");
            g_tui->record_request_start();
        }

        const auto t0 = std::chrono::high_resolution_clock::now();
        const std::string tools_json = build_tools_json(tools);

        std::string cmpl_id = "chatcmpl-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());

        if (stream) {
            std::string header = "HTTP/1.1 200 OK\r\n"
                                 "Content-Type: text/event-stream; charset=utf-8\r\n"
                                 "Cache-Control: no-cache\r\n"
                                 "Connection: keep-alive\r\n"
                                 "Access-Control-Allow-Origin: *\r\n\r\n";
            write(client_fd, header.c_str(), header.size());

            // 1. Initial role chunk
            std::string chunk0 = "data: {\"id\":\"" + cmpl_id + "\",\"object\":\"chat.completion.chunk\",\"created\":1787300000,\"model\":\"" + model_requested + "\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}\n\n";
            write(client_fd, chunk0.c_str(), chunk0.size());

            // 2. Real generation. Reasoning and content deltas come from the
            // model itself via ThinkSplitter; tool calls are parsed from the
            // model's <tool_call> output after the turn completes.
            size_t token_count = 0;
            bool got_first_token = false;
            double ttft_ms = 0.0;
            std::string full_content;
            auto t_first_token = t0;

            auto emit_sse = [&](const std::string& json_delta) {
                std::string c = "data: {\"id\":\"" + cmpl_id + "\",\"object\":\"chat.completion.chunk\",\"created\":1787300000,\"model\":\"" + model_requested + "\",\"choices\":[{\"index\":0,\"delta\":{" + json_delta + "},\"finish_reason\":null}]}\n\n";
                write(client_fd, c.c_str(), c.size());
            };

            ThinkSplitter splitter(
                [&](const std::string& thought) {
                    if (enable_thinking && !thought.empty())
                        emit_sse("\"reasoning_content\":\"" + json_escape(thought) + "\"");
                },
                [&](const std::string& text) {
                    // Once the model opens a <tool_call> block, suppress raw
                    // content deltas; the structured tool_calls delta at the end
                    // carries it instead.
                    full_content += text;
                    if (full_content.find("<tool_call>") == std::string::npos)
                        emit_sse("\"content\":\"" + json_escape(text) + "\"");
                },
                enable_thinking);

            if (engine) {
                std::lock_guard<std::mutex> lock(g_engine_mutex);
                if (g_tui) g_tui->log("Prefilling " + std::to_string(formatted_msgs.size()) + " messages on 64 ANE layers...", "ENGINE");
                auto stream_token_cb = [&](const std::string& token_chunk) {
                    if (token_chunk.empty()) return;
                    ++token_count;
                    if (!got_first_token) {
                        got_first_token = true;
                        t_first_token = std::chrono::high_resolution_clock::now();
                        ttft_ms = std::chrono::duration<double, std::milli>(t_first_token - t0).count();
                        if (g_tui) g_tui->log("Prefill complete (TTFT: " + std::to_string((int)ttft_ms) + "ms). Streaming tokens...", "ENGINE");
                    }
                    splitter.feed(token_chunk);
                    if (g_tui) g_tui->record_request_chunk(1);
                };
                engine->chat_completion(formatted_msgs, tools_json,
                                        requested_max_tokens, requested_temperature,
                                        stream_token_cb, enable_thinking,
                                        reasoning_effort);
            }
            splitter.finish();

            std::string clean_content;
            std::vector<ParsedToolCall> tool_calls;
            parse_qwen_tool_calls(full_content, clean_content, tool_calls);
            const bool has_tool_calls = !tool_calls.empty();
            for (const auto& tc : tool_calls) {
                emit_sse("\"tool_calls\":[{\"index\":0,\"id\":\"" + tc.id +
                         "\",\"type\":\"function\",\"function\":{\"name\":\"" +
                         json_escape(tc.name) + "\",\"arguments\":\"\"}}]");
                emit_sse("\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"" +
                         json_escape(tc.arguments_json) + "\"}}]");
            }

            // Final chunk with finish_reason and REAL usage counters.
            const GenerationStats empty_stats;
            const GenerationStats& stats = (engine && engine->is_ready())
                ? engine->get_last_stats() : empty_stats;
            std::string final_chunk = "data: {\"id\":\"" + cmpl_id + "\",\"object\":\"chat.completion.chunk\",\"created\":1787300000,\"model\":\"" + model_requested + "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"" + (has_tool_calls ? "tool_calls" : "stop") + "\"}],\"usage\":{\"prompt_tokens\":" + std::to_string(stats.prompt_tokens) + ",\"completion_tokens\":" + std::to_string(stats.generated_tokens) + ",\"total_tokens\":" + std::to_string(stats.prompt_tokens + stats.generated_tokens) + "}}\n\n";
            write(client_fd, final_chunk.c_str(), final_chunk.size());

            // 5. DONE marker
            std::string done_marker = "data: [DONE]\n\n";
            write(client_fd, done_marker.c_str(), done_marker.size());

            shutdown(client_fd, SHUT_WR);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            close(client_fd);

            const GenerationStats& sstats = (engine && engine->is_ready())
                ? engine->get_last_stats() : empty_stats;
            auto t_end = std::chrono::high_resolution_clock::now();
            double total_decode_sec = got_first_token
                ? std::chrono::duration<double>(t_end - t_first_token).count() : 0.0;
            double decode_tps = (token_count > 1 && total_decode_sec > 0.0)
                ? static_cast<double>(token_count - 1) / total_decode_sec : 0.0;
            const size_t real_prompt_tokens = sstats.prompt_tokens;

            if (g_tui) {
                g_tui->record_request_end(real_prompt_tokens, token_count, ttft_ms,
                                          decode_tps, sstats.prefill_tps(),
                                          g_engine->last_apc_hit(),
                                          g_engine->last_apc_saved(), true);
                char lbuf[160];
                snprintf(lbuf, sizeof(lbuf),
                         "Stream complete: %zu tokens in %.2fs (%.1f tok/s) [TTFT: %.1fms, prefill %zu tok @ %.1f tok/s]",
                         token_count, total_decode_sec, decode_tps, ttft_ms,
                         real_prompt_tokens, sstats.prefill_tps());
                g_tui->log(std::string(lbuf), "ENGINE");
            }
            return;
        } else {
            // Non-streaming completion response over the real model.
            std::string gen_output;
            if (engine) {
                std::lock_guard<std::mutex> lock(g_engine_mutex);
                gen_output = engine->chat_completion(formatted_msgs, tools_json,
                                                     requested_max_tokens,
                                                     requested_temperature, nullptr,
                                                     enable_thinking, reasoning_effort);
            }
            if (gen_output.empty()) {
                std::string err_body = "{\"error\":{\"message\":\"native inference produced no output\",\"type\":\"server_error\"}}";
                send_json_response(client_fd, 500, "Internal Server Error", err_body);
                close(client_fd);
                return;
            }

            // Split the real reasoning/content and extract tool calls.
            ThinkSplitter offline(
                [](const std::string&) {},
                [](const std::string&) {},
                enable_thinking);
            offline.feed(gen_output);
            offline.finish();
            std::string clean_content;
            std::vector<ParsedToolCall> tool_calls;
            parse_qwen_tool_calls(offline.content_accum(), clean_content, tool_calls);
            const bool has_tool_calls = !tool_calls.empty();
            const std::string& reasoning_content = offline.reasoning_accum();
            const std::string& content = clean_content;

            const GenerationStats empty_stats_ns;
            const GenerationStats& stats = (engine && engine->is_ready())
                ? engine->get_last_stats() : empty_stats_ns;
            auto t_first_token = t0;  // non-streaming: first byte arrives with everything else

            std::string message_fields;
            if (!has_tool_calls) {
                message_fields = "\"content\":" + (content.empty() ? std::string("null") : "\"" + json_escape(content) + "\"");
                if (!reasoning_content.empty())
                    message_fields += ",\"reasoning_content\":\"" + json_escape(reasoning_content) + "\"";
            } else {
                message_fields = "\"content\":" + (content.empty() ? std::string("null") : "\"" + json_escape(content) + "\"");
                if (!reasoning_content.empty())
                    message_fields += ",\"reasoning_content\":\"" + json_escape(reasoning_content) + "\"";
                message_fields += ",\"tool_calls\":[";
                for (size_t i = 0; i < tool_calls.size(); ++i) {
                    if (i) message_fields += ",";
                    message_fields += "{\"id\":\"" + json_escape(tool_calls[i].id) + "\",\"type\":\"function\",\"function\":{\"name\":\"" + json_escape(tool_calls[i].name) + "\",\"arguments\":\"" + json_escape(tool_calls[i].arguments_json) + "\"}}";
                }
                message_fields += "]";
            }
            const std::string finish_reason = has_tool_calls ? "tool_calls" : "stop";

            std::string res_body = "{\"id\":\"" + cmpl_id + "\",\"object\":\"chat.completion\",\"created\":1787300000,\"model\":\"" + model_requested + "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\"," + message_fields + "},\"finish_reason\":\"" + finish_reason + "\"}],\"usage\":{\"prompt_tokens\":" + std::to_string(stats.prompt_tokens) + ",\"completion_tokens\":" + std::to_string(stats.generated_tokens) + ",\"total_tokens\":" + std::to_string(stats.prompt_tokens + stats.generated_tokens) + "}}";
            send_json_response(client_fd, 200, "OK", res_body);
            close(client_fd);

            auto t_end = std::chrono::high_resolution_clock::now();
            double ttft_ms = std::chrono::duration<double, std::milli>(t_end - t_first_token).count();
            const double decode_tps = stats.decode_tps();

            if (g_tui) {
                g_tui->record_request_end(stats.prompt_tokens, stats.generated_tokens,
                                          ttft_ms, decode_tps, stats.prefill_tps(),
                                          g_engine->last_apc_hit(),
                                          g_engine->last_apc_saved());
                char lbuf[160];
                snprintf(lbuf, sizeof(lbuf),
                         "Non-stream complete: %zu tokens in %.2fs (%.1f tok/s) [TTFT: %.1fms, prefill %zu tok @ %.1f tok/s]",
                         stats.generated_tokens, stats.total_ms / 1000.0, decode_tps, ttft_ms,
                         stats.prompt_tokens, stats.prefill_tps());
                g_tui->log(std::string(lbuf), "ENGINE");
            }
            return;
        }
    }

    std::string resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    write(client_fd, resp.c_str(), resp.size());
    close(client_fd);
}

int main(int argc, char* argv[]) {
    setenv("RINDI_TAIL_COREAI", "1", 0);
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    ServerConfig config;
    config.port = 2456;
    config.host = "0.0.0.0";
    config.model_name = "Qwen3.8-27B";
    config.device_name = "Apple M5 Max";
    config.mode = "silent";
    config.resident_layers = 64;

    std::string model_path = "/Users/true/.lmstudio/models/Qwen/Qwen3.8-27B.rindi";
    bool chat_only = false;

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "chat" || std::string(argv[i]) == "--chat") {
            chat_only = true;
        } else if (std::string(argv[i]) == "--port" && i + 1 < argc) {
            config.port = std::stoi(argv[i + 1]);
        } else if (std::string(argv[i]) == "--host" && i + 1 < argc) {
            config.host = argv[i + 1];
        } else if (std::string(argv[i]) == "--mode" && i + 1 < argc) {
            config.mode = argv[i + 1];
        } else if (std::string(argv[i]) == "--model" && i + 1 < argc) {
            model_path = argv[i + 1];
            config.model_name = argv[i + 1];
        } else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            std::cout << "Rindi Apple Silicon Standalone Native C++ ANE + Metal GPU Inference Engine\n"
                      << "Usage: rindi [chat] [options]\n\n"
                      << "Commands:\n"
                      << "  chat                Open the focused local chat interface\n\n"
                      << "Options:\n"
                      << "  --port <port>       Port to listen on (current: " << config.port << ")\n"
                      << "  --host <host>       Host IP to bind to (current: " << config.host << ")\n"
                      << "  --model <path>      Path to 27B model (current: " << model_path << ")\n"
                      << "  --mode <mode>       Engine mode: silent or turbo (default: silent)\n"
                      << "  --help, -h          Show this help message\n";
            return 0;
        }
    }

    // 1. Initialize Rindi TUI Controller
    g_tui = new RindiTUI(config);

    // 2. Initialize Pure C++ Native 27B Engine
    g_tui->log("Initializing 27B Model Engine on ANE + Metal GPU...", "ENGINE");
    g_engine = new RindiEngine(model_path);
    if (!g_engine->is_ready()) {
        g_tui->log("Native transformer scheduler failed; refusing to advertise an inference endpoint", "ERROR");
        delete g_engine;
        g_engine = nullptr;
        g_tui->stop();
        delete g_tui;
        g_tui = nullptr;
        return 1;
    }
    g_tui->log("Loaded 27B Model (" + model_path + ") with 64 ANE Layers", "ANE");

    // Focused local conversation mode. It deliberately skips socket setup and
    // the telemetry dashboard: one process owns the model, conversation state,
    // and APC cache, while token deltas stream directly to the terminal.
    if (chat_only) {
        const int result = run_local_chat(g_engine, config);
        delete g_engine;
        g_engine = nullptr;
        delete g_tui;
        g_tui = nullptr;
        return result;
    }

    // 3. Setup High-Performance POSIX Socket Server
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    bool socket_ok = false;
    std::string socket_error;
    if (server_fd < 0) {
        socket_error = "socket: " + std::string(std::strerror(errno));
    }
    if (server_fd >= 0) {
        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in address{};
        address.sin_family = AF_INET;
        if (config.host == "0.0.0.0" || config.host.empty()) {
            address.sin_addr.s_addr = htonl(INADDR_ANY);
        } else if (inet_pton(AF_INET, config.host.c_str(),
                             &address.sin_addr) != 1) {
            socket_error = "invalid IPv4 bind address: " + config.host;
            close(server_fd);
            server_fd = -1;
        }
        address.sin_port = htons(config.port);

        if (server_fd >= 0 &&
            bind(server_fd, (struct sockaddr*)&address, sizeof(address)) >= 0) {
            if (listen(server_fd, 128) >= 0) {
                socket_ok = true;
                g_tui->log("HTTP Server listening on " + config.host + ":" + std::to_string(config.port), "HTTP");
                g_tui->log("OpenAI API endpoint active: /v1/chat/completions, /v1/models", "INFO");
                std::cout << "[Rindi] HTTP server ready at http://" << config.host << ':'
                          << config.port << " (OpenAI API: /v1)\n" << std::flush;
            } else {
                socket_error = "listen: " + std::string(std::strerror(errno));
            }
        } else if (server_fd >= 0) {
            socket_error = "bind " + config.host + ":" + std::to_string(config.port) +
                           ": " + std::string(std::strerror(errno));
        }
    }
    if (!socket_ok) {
        if (server_fd >= 0) {
            close(server_fd);
            server_fd = -1;
        }
        if (socket_error.empty()) socket_error = "unknown socket error";
        std::cerr << "[Rindi] HTTP server failed: " << socket_error << '\n' << std::flush;
        g_tui->log("HTTP server failed: " + socket_error, "ERROR");

        // A headless process with no listening socket is unusable and otherwise
        // appears to have launched successfully while sleeping forever.
        if (std::getenv("RINDI_HEADLESS") != nullptr) {
            delete g_engine;
            g_engine = nullptr;
            delete g_tui;
            g_tui = nullptr;
            return 1;
        }
    }

    const bool headless = std::getenv("RINDI_HEADLESS") != nullptr;

    // 4. Start Background Real-time TUI Renderer (4 Hz) unless the process is
    // being run as a service.  Headless mode keeps native HTTP deployments
    // from continuously repainting a terminal and makes stderr diagnostics
    // usable by launchd/systemd-style supervisors.
    if (!headless) g_tui->start_renderer(4);

    // 5. Start Background Socket Accept Thread if socket online
    std::thread server_thread;
    if (socket_ok) {
        server_thread = std::thread([server_fd]() {
            while (g_running.load()) {
                sockaddr_in client_addr{};
                socklen_t client_len = sizeof(client_addr);
                int client_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
                if (client_fd < 0) {
                    if (!g_running.load()) break;
                    continue;
                }
                std::thread(handle_client, client_fd, g_engine).detach();
            }
        });
    }

    // 6. Interactive Command & Chat Loop on Main Thread, or a simple service
    // loop when launched headlessly.
    auto chat_dispatch = [](const std::string& prompt, std::function<void(const std::string& token)> stream_cb) -> RindiTUI::ChatTurnStats {
        RindiTUI::ChatTurnStats stats;
        if (g_engine) {
            std::vector<std::pair<std::string, std::string>> msgs = {{"user", prompt}};
            g_engine->chat_completion(msgs, "", 128, 0.7f, stream_cb, false);
            const GenerationStats& s = g_engine->get_last_stats();
            stats.prompt_tokens = s.prompt_tokens;
            stats.generated_tokens = s.generated_tokens;
            stats.ttft_ms = s.ttft_ms;
            stats.prefill_tps = s.prefill_tps();
            stats.decode_tps = s.decode_tps();
        }
        return stats;
    };

    if (headless) {
        while (g_running.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    } else {
        g_tui->run_interactive_loop(nullptr, chat_dispatch);
    }

    // Shutdown sequence
    g_running.store(false);
    if (!headless) g_tui->stop_renderer();

    if (socket_ok) {
        shutdown(server_fd, SHUT_RDWR);
        close(server_fd);
        if (server_thread.joinable()) {
            server_thread.join();
        }
    }

    delete g_engine;
    delete g_tui;

    std::cout << "\n[Rindi] Server stopped gracefully.\n" << std::flush;
    return 0;
}
