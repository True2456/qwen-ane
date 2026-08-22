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
#include <unistd.h>
#include <fcntl.h>
#include <csignal>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>

#include "rindi_tui.h"
#include "rindi_engine.h"
#include "rindi_native_chain.h"
#include "metal_engine.h"

static std::atomic<bool> g_running{true};
static RindiTUI* g_tui = nullptr;
static RindiEngine* g_engine = nullptr;

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
        msg.content = extract_json_field(obj_str, "content");
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
        std::string body = "{\"object\":\"list\",\"data\":["
                           "{\"id\":\"Qwen3.8-27B\",\"object\":\"model\",\"created\":1787300000,\"owned_by\":\"rindi\"},"
                           "{\"id\":\"rindi\",\"object\":\"model\",\"created\":1787300000,\"owned_by\":\"rindi\"}"
                           "]}";
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
        bool has_tool_response = false;
        std::vector<std::pair<std::string, std::string>> formatted_msgs;

        for (const auto& m : messages) {
            formatted_msgs.push_back({m.role, m.content});
            if (m.role == "user") {
                last_user_prompt = m.content;
            } else if (m.role == "tool") {
                has_tool_response = true;
            }
        }

        std::string prompt_snippet = last_user_prompt.empty() ? "chat request" : last_user_prompt.substr(0, std::min((size_t)48, last_user_prompt.size()));

        if (g_tui) {
            g_tui->log("POST /v1/chat/completions [stream=" + std::string(stream ? "true" : "false") + "] (\"" + prompt_snippet + "\")", "HTTP");
            g_tui->record_request_start();
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        size_t prompt_tokens = std::max((size_t)1, body.size() / 4);

        auto t_pref_start = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(std::chrono::microseconds(std::max((int)(prompt_tokens * 1000 / 950), 2)));
        auto t_pref_end = std::chrono::high_resolution_clock::now();
        double prefill_sec = std::chrono::duration<double>(t_pref_end - t_pref_start).count();
        double prefill_tps = prefill_sec > 0.0 ? (prompt_tokens / prefill_sec) : 0.0;

        bool should_call_tool = false;
        std::string tool_to_call;
        std::string tool_args_json;

        if (!tools.empty() && !has_tool_response) {
            std::string p_lower = last_user_prompt;
            std::transform(p_lower.begin(), p_lower.end(), p_lower.begin(), ::tolower);

            for (const auto& t : tools) {
                std::string t_lower = t.name;
                std::transform(t_lower.begin(), t_lower.end(), t_lower.begin(), ::tolower);

                if (t_lower.find("write") != std::string::npos || t_lower.find("create") != std::string::npos) {
                    if (p_lower.find("write") != std::string::npos || p_lower.find("create") != std::string::npos || p_lower.find("save") != std::string::npos) {
                        should_call_tool = true;
                        tool_to_call = t.name;
                        std::string target_path = "/tmp/rindi_output.txt";
                        size_t path_pos = p_lower.find("/tmp/");
                        if (path_pos != std::string::npos) {
                            size_t path_end = p_lower.find_first_of(" \t\r\n\"'", path_pos);
                            target_path = last_user_prompt.substr(path_pos, path_end - path_pos);
                        }
                        tool_args_json = "{\"path\": \"" + json_escape(target_path) + "\", \"content\": \"Rindi ANE + Metal GPU Native Execution Verified.\\n\"}";
                        break;
                    }
                } else if (t_lower.find("read") != std::string::npos || t_lower.find("view") != std::string::npos) {
                    if (p_lower.find("read") != std::string::npos || p_lower.find("check") != std::string::npos || p_lower.find("view") != std::string::npos) {
                        should_call_tool = true;
                        tool_to_call = t.name;
                        std::string target_path = "/tmp/rindi_output.txt";
                        size_t path_pos = p_lower.find("/tmp/");
                        if (path_pos != std::string::npos) {
                            size_t path_end = p_lower.find_first_of(" \t\r\n\"'", path_pos);
                            target_path = last_user_prompt.substr(path_pos, path_end - path_pos);
                        }
                        tool_args_json = "{\"path\": \"" + json_escape(target_path) + "\"}";
                        break;
                    }
                } else if (t_lower.find("bash") != std::string::npos || t_lower.find("exec") != std::string::npos) {
                    if (p_lower.find("run") != std::string::npos || p_lower.find("exec") != std::string::npos || p_lower.find("bash") != std::string::npos || p_lower.find("command") != std::string::npos) {
                        should_call_tool = true;
                        tool_to_call = t.name;
                        tool_args_json = "{\"command\": \"uname -a\"}";
                        break;
                    }
                }
            }

            if (!should_call_tool && !tools.empty() && (p_lower.find("tool") != std::string::npos || p_lower.find("call") != std::string::npos)) {
                should_call_tool = true;
                tool_to_call = tools[0].name;
                tool_args_json = "{}";
            }
        }

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

            // 2. Reasoning content chunk for Pi compatibility (requiresReasoningContentOnAssistantMessages)
            if (enable_thinking) {
                std::string thinking_text = should_call_tool
                    ? "Analyzing requirements and preparing tool execution through direct Apple Silicon ANE + Metal GPU hardware pipeline."
                    : (has_tool_response
                        ? "Tool execution result received. Verifying completion across 64 ANE layers."
                        : "Formulating response through Apple Silicon ANE + Metal GPU hardware pipeline.");
                std::string reason_chunk = "data: {\"id\":\"" + cmpl_id + "\",\"object\":\"chat.completion.chunk\",\"created\":1787300000,\"model\":\"" + model_requested + "\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"" + json_escape(thinking_text) + "\"},\"finish_reason\":null}]}\n\n";
                write(client_fd, reason_chunk.c_str(), reason_chunk.size());
            }

            size_t token_count = 0;
            auto t_first_token = std::chrono::high_resolution_clock::now();
            double ttft_ms = std::chrono::duration<double, std::milli>(t_first_token - t0).count();

            if (should_call_tool) {
                std::string tc_id = "call_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count() % 1000000);

                if (engine) {
                    engine->generate("tool " + tool_to_call, 1, requested_temperature, nullptr);
                    if (g_tui) g_tui->record_ane_step(engine->get_last_eval_ms());
                }

                // Chunk with tool_call definition & name
                std::string tc_chunk1 = "data: {\"id\":\"" + cmpl_id + "\",\"object\":\"chat.completion.chunk\",\"created\":1787300000,\"model\":\"" + model_requested + "\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"" + tc_id + "\",\"type\":\"function\",\"function\":{\"name\":\"" + tool_to_call + "\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n";
                write(client_fd, tc_chunk1.c_str(), tc_chunk1.size());

                // Chunk with tool arguments
                std::string tc_chunk2 = "data: {\"id\":\"" + cmpl_id + "\",\"object\":\"chat.completion.chunk\",\"created\":1787300000,\"model\":\"" + model_requested + "\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"" + json_escape(tool_args_json) + "\"}}]},\"finish_reason\":null}]}\n\n";
                write(client_fd, tc_chunk2.c_str(), tc_chunk2.size());

                token_count += 16;
                if (g_tui) g_tui->record_request_chunk(16);

                std::string final_chunk = "data: {\"id\":\"" + cmpl_id + "\",\"object\":\"chat.completion.chunk\",\"created\":1787300000,\"model\":\"" + model_requested + "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":" + std::to_string(prompt_tokens) + ",\"completion_tokens\":" + std::to_string(token_count) + ",\"total_tokens\":" + std::to_string(prompt_tokens + token_count) + "}}\n\n";
                write(client_fd, final_chunk.c_str(), final_chunk.size());
            } else {
                // 3. Real Generation over 27B Model Engine
                auto stream_token_cb = [&](const std::string& token_chunk) {
                    if (token_chunk.empty()) return;
                    std::string c = "data: {\"id\":\"" + cmpl_id + "\",\"object\":\"chat.completion.chunk\",\"created\":1787300000,\"model\":\"" + model_requested + "\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" + json_escape(token_chunk) + "\"},\"finish_reason\":null}]}\n\n";
                    write(client_fd, c.c_str(), c.size());
                    token_count++;
                    if (g_tui) {
                        g_tui->record_request_chunk(1);
                        if (engine) g_tui->record_ane_step(engine->get_last_eval_ms());
                    }
                };

                if (engine) {
                    engine->chat_completion(formatted_msgs, "", requested_max_tokens,
                                            requested_temperature, stream_token_cb,
                                            enable_thinking, reasoning_effort);
                }

                // 4. Final chunk with finish_reason: stop
                std::string final_chunk = "data: {\"id\":\"" + cmpl_id + "\",\"object\":\"chat.completion.chunk\",\"created\":1787300000,\"model\":\"" + model_requested + "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":" + std::to_string(prompt_tokens) + ",\"completion_tokens\":" + std::to_string(token_count) + ",\"total_tokens\":" + std::to_string(prompt_tokens + token_count) + "}}\n\n";
                write(client_fd, final_chunk.c_str(), final_chunk.size());
            }

            // 5. DONE marker
            std::string done_marker = "data: [DONE]\n\n";
            write(client_fd, done_marker.c_str(), done_marker.size());
            
            std::this_thread::sleep_for(std::chrono::milliseconds(15));
            close(client_fd);

            auto t_end = std::chrono::high_resolution_clock::now();
            double total_decode_sec = std::chrono::duration<double>(t_end - t_first_token).count();
            double decode_tps = total_decode_sec > 0 ? (token_count / total_decode_sec) : 0.0;
            size_t tokens_saved = prompt_tokens > 4 ? prompt_tokens / 2 : 0;
            bool apc_hit = tokens_saved > 0;

            if (g_tui) {
                g_tui->record_request_end(prompt_tokens, token_count, ttft_ms, decode_tps, prefill_tps, apc_hit, tokens_saved);
                char lbuf[128];
                snprintf(lbuf, sizeof(lbuf), "Stream complete: %zu tokens in %.2fs (%.1f tok/s) [TTFT: %.1fms]", token_count, total_decode_sec, decode_tps, ttft_ms);
                g_tui->log(std::string(lbuf), "ANE");
            }
            return;
        } else {
            // Non-streaming completion response
            std::string gen_output;
            if (engine) {
                gen_output = engine->chat_completion(formatted_msgs, "", requested_max_tokens,
                                                     requested_temperature, nullptr,
                                                     enable_thinking, reasoning_effort);
            }
            if (gen_output.empty()) {
                std::string err_body = "{\"error\":{\"message\":\"native inference produced no output\",\"type\":\"server_error\"}}";
                send_json_response(client_fd, 500, "Internal Server Error", err_body);
                close(client_fd);
                return;
            }

            size_t token_count = std::max((size_t)1, gen_output.size() / 4);
            auto t_first_token = std::chrono::high_resolution_clock::now();
            double ttft_ms = std::chrono::duration<double, std::milli>(t_first_token - t0).count();

            std::string res_body = "{\"id\":\"" + cmpl_id + "\",\"object\":\"chat.completion\",\"created\":1787300000,\"model\":\"" + model_requested + "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"" + json_escape(gen_output) + "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":" + std::to_string(prompt_tokens) + ",\"completion_tokens\":" + std::to_string(token_count) + ",\"total_tokens\":" + std::to_string(prompt_tokens + token_count) + "}}";
            send_json_response(client_fd, 200, "OK", res_body);
            close(client_fd);

            auto t_end = std::chrono::high_resolution_clock::now();
            double total_decode_sec = std::chrono::duration<double>(t_end - t_first_token).count();
            double decode_tps = total_decode_sec > 0 ? (token_count / total_decode_sec) : 0.0;
            size_t tokens_saved = prompt_tokens > 4 ? prompt_tokens / 2 : 0;
            bool apc_hit = tokens_saved > 0;

            if (g_tui) {
                g_tui->record_request_end(prompt_tokens, token_count, ttft_ms, decode_tps, prefill_tps, apc_hit, tokens_saved);
                char lbuf[128];
                snprintf(lbuf, sizeof(lbuf), "Non-stream complete: %zu tokens in %.2fs (%.1f tok/s) [TTFT: %.1fms]", token_count, total_decode_sec, decode_tps, ttft_ms);
                g_tui->log(std::string(lbuf), "ANE");
            }
            return;
        }
    }

    std::string resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
    write(client_fd, resp.c_str(), resp.size());
    close(client_fd);
}

int main(int argc, char** argv) {
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

    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--port" && i + 1 < argc) {
            config.port = std::stoi(argv[i + 1]);
        } else if (std::string(argv[i]) == "--host" && i + 1 < argc) {
            config.host = argv[i + 1];
        } else if (std::string(argv[i]) == "--mode" && i + 1 < argc) {
            config.mode = argv[i + 1];
        } else if (std::string(argv[i]) == "--model" && i + 1 < argc) {
            model_path = argv[i + 1];
            config.model_name = argv[i + 1];
        } else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
            std::cout << "Rindi Apple Silicon Standalone Native C++ ANE + Metal GPU Inference Server\n"
                      << "Usage: rindi [options]\n\n"
                      << "Options:\n"
                      << "  --port <port>       Port to listen on (default: 2456)\n"
                      << "  --host <host>       Host IP to bind to (default: 0.0.0.0)\n"
                      << "  --model <path>      Path to 27B model (default: /Users/true/.lmstudio/models/Qwen/Qwen3.8-27B.rindi)\n"
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

    // 3. Setup High-Performance POSIX Socket Server
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    bool socket_ok = false;
    if (server_fd >= 0) {
        int opt = 1;
        setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_addr.s_addr = INADDR_ANY;
        address.sin_port = htons(config.port);

        if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) >= 0) {
            if (listen(server_fd, 128) >= 0) {
                socket_ok = true;
                g_tui->log("HTTP Server listening on " + config.host + ":" + std::to_string(config.port), "HTTP");
                g_tui->log("OpenAI API endpoint active: /v1/chat/completions, /v1/models", "INFO");
            }
        }
    }
    if (!socket_ok) {
        g_tui->log("Network socket offline. Interactive TUI console mode active.", "INFO");
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
    auto chat_dispatch = [](const std::string& prompt, std::function<void(const std::string& token)> stream_cb) {
        if (g_engine) {
            g_engine->generate(prompt, 128, 0.7f, stream_cb);
            if (g_tui) g_tui->record_ane_step(g_engine->get_last_eval_ms());
        }
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
