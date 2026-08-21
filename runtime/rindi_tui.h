/*
 * SPDX-License-Identifier: Apache-2.0
 * runtime/rindi_tui.h - High-Performance Real-Time Terminal User Interface for Rindi Native Server.
 */

#ifndef RINDI_TUI_H
#define RINDI_TUI_H

#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <cstdint>

// ANSI Color and Style Definitions (Zero Emojis)
namespace RindiANSI {
    constexpr const char* RESET       = "\033[0m";
    constexpr const char* BOLD        = "\033[1m";
    constexpr const char* DIM         = "\033[2m";
    constexpr const char* ITALIC      = "\033[3m";
    constexpr const char* UNDERLINE   = "\033[4m";
    constexpr const char* INVERSE     = "\033[7m";

    constexpr const char* FG_BLACK    = "\033[30m";
    constexpr const char* FG_RED      = "\033[31m";
    constexpr const char* FG_GREEN    = "\033[32m";
    constexpr const char* FG_YELLOW   = "\033[33m";
    constexpr const char* FG_BLUE     = "\033[34m";
    constexpr const char* FG_MAGENTA  = "\033[35m";
    constexpr const char* FG_CYAN     = "\033[36m";
    constexpr const char* FG_WHITE    = "\033[37m";

    constexpr const char* FG_BRIGHT_BLACK   = "\033[90m";
    constexpr const char* FG_BRIGHT_RED     = "\033[91m";
    constexpr const char* FG_BRIGHT_GREEN   = "\033[92m";
    constexpr const char* FG_BRIGHT_YELLOW  = "\033[93m";
    constexpr const char* FG_BRIGHT_BLUE    = "\033[94m";
    constexpr const char* FG_BRIGHT_MAGENTA = "\033[95m";
    constexpr const char* FG_BRIGHT_CYAN    = "\033[96m";
    constexpr const char* FG_BRIGHT_WHITE   = "\033[97m";

    constexpr const char* CLEAR_SCREEN = "\033[H\033[J";
    constexpr const char* CURSOR_HOME  = "\033[H";
    constexpr const char* CURSOR_HIDE  = "\033[?25l";
    constexpr const char* CURSOR_SHOW  = "\033[?25h";
    constexpr const char* ERASE_LINE   = "\033[2K";
}

struct InferenceMetrics {
    std::atomic<uint64_t> total_requests{0};
    std::atomic<uint64_t> active_requests{0};
    std::atomic<uint64_t> total_prompt_tokens{0};
    std::atomic<uint64_t> total_completion_tokens{0};
    std::atomic<uint64_t> total_tokens_saved{0};

    std::atomic<double> last_ttft_ms{0.0};
    std::atomic<double> avg_ttft_ms{0.0};
    std::atomic<double> last_decode_tps{0.0};
    std::atomic<double> avg_decode_tps{0.0};
    std::atomic<double> last_prefill_tps{0.0};
    std::atomic<double> last_ane_latency_ms{0.0};

    std::atomic<uint64_t> apc_hits{0};
    std::atomic<uint64_t> apc_misses{0};
    std::atomic<bool> last_apc_hit{false};
};

struct HardwareMetrics {
    uint64_t process_rss_bytes{0};
    uint64_t process_virt_bytes{0};
    uint64_t system_total_ram_bytes{0};
    uint64_t system_used_ram_bytes{0};
    uint64_t model_blobs_bytes{13087227904ULL}; // ~12.19 GB
    uint64_t host_ram_freed_bytes{44023414784ULL}; // ~41.0 GB

    double soc_power_w{5.9};
    double ane_power_w{4.8};
    double gpu_power_w{0.3};
    double cpu_power_w{0.8};
};

struct ServerConfig {
    std::string host{"0.0.0.0"};
    int port{2456};
    std::string model_name{"Qwen3.8-27B"};
    std::string device_name{"Apple M5 Max"};
    std::string mode{"silent"}; // "silent" or "turbo"
    size_t resident_layers{64};
    size_t hidden_dim{5120};
    size_t seq_len{32};
    float temperature{0.7f};
    int max_tokens{128};
    float top_p{0.9f};
};

class RindiTUI {
public:
    using CommandHandler = std::function<std::string(const std::string& cmd, const std::string& args)>;
    using ChatDispatchFn = std::function<void(const std::string& prompt, std::function<void(const std::string& token)> stream_cb)>;

    RindiTUI(const ServerConfig& config = ServerConfig());
    ~RindiTUI();

    // Logging & Event Stream
    void log(const std::string& message, const std::string& tag = "INFO");

    // Metrics recording
    void record_request_start();
    void record_request_chunk(size_t token_count = 1);
    void record_request_end(size_t prompt_tokens, size_t completion_tokens,
                           double ttft_ms, double decode_tps, double prefill_tps,
                           bool apc_hit, size_t tokens_saved);
    void record_ane_step(double latency_ms);

    // Dashboard rendering
    std::string render_dashboard();
    void refresh_display();

    // Background auto-refresh loop
    void start_renderer(int refresh_rate_hz = 4);
    void stop_renderer();

    // Interactive command loop
    void run_interactive_loop(CommandHandler cmd_handler = nullptr, ChatDispatchFn chat_fn = nullptr);

    // Hardware polling
    void update_hardware_metrics();

    // Config access
    ServerConfig& config() { return config_; }
    InferenceMetrics& metrics() { return metrics_; }
    HardwareMetrics& hardware() { return hardware_; }

    void set_mode(const std::string& mode);
    bool is_running() const { return running_.load(); }
    void stop() { running_.store(false); }

private:
    ServerConfig config_;
    InferenceMetrics metrics_;
    HardwareMetrics hardware_;

    std::deque<std::string> log_buffer_;
    std::deque<std::pair<std::string, std::string>> chat_history_; // {user_prompt, assistant_resp}
    size_t max_logs_{200};
    std::mutex log_mutex_;
    std::mutex render_mutex_;

    std::atomic<bool> running_{true};
    std::atomic<bool> in_chat_stream_{false};
    std::string current_chat_stream_chunk_;
    std::thread renderer_thread_;

    std::string current_input_line_;
    std::mutex input_mutex_;

    std::string format_bytes(uint64_t bytes) const;
    std::string progress_bar(double fraction, int width, const std::string& color) const;
};

#endif // RINDI_TUI_H
