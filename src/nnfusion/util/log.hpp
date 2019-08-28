// Microsoft (c) 2019, Wenxiang Hu
// This file is modified from nnfusion Log
#pragma once

#include <deque>
#include <functional>
#include <sstream>
#include <stdexcept>

#ifndef PROJECT_ROOT_DIR
#define PROJECT_ROOT_DIR "NNFusion"
#endif

namespace nnfusion
{
    class ConstString
    {
    public:
        template <size_t SIZE>
        constexpr ConstString(const char (&p)[SIZE])
            : m_string(p)
            , m_size(SIZE)
        {
        }

        constexpr char operator[](size_t i) const
        {
            return i < m_size ? m_string[i] : throw std::out_of_range("");
        }
        constexpr const char* get_ptr(size_t offset) const { return &m_string[offset]; }
        constexpr size_t size() const { return m_size; }
    private:
        const char* m_string;
        size_t m_size;
    };

    constexpr const char* find_last(ConstString s, size_t offset, char ch)
    {
        return offset == 0 ? s.get_ptr(0) : (s[offset] == ch ? s.get_ptr(offset + 1)
                                                             : find_last(s, offset - 1, ch));
    }

    constexpr const char* find_last(ConstString s, char ch)
    {
        return find_last(s, s.size() - 1, ch);
    }

    constexpr const char* get_file_name(ConstString s) { return find_last(s, '/'); }
    constexpr const char* trim_file_name(ConstString root, ConstString s)
    {
        return s.get_ptr(root.size());
    }
    enum class LOG_TYPE
    {
        _LOG_TYPE_ERROR,
        _LOG_TYPE_WARNING,
        _LOG_TYPE_INFO,
        _LOG_TYPE_DEBUG,
    };

    class LogHelper
    {
    public:
        LogHelper(LOG_TYPE,
                  const char* file,
                  int line,
                  std::function<void(const std::string&)> m_handler_func);
        ~LogHelper();

        std::ostream& stream() { return m_stream; }
        static void set_log_path(const std::string& path)
        {
            flag_save_to_file = true;
            log_path = path;
        }

    private:
        std::function<void(const std::string&)> m_handler_func;
        std::stringstream m_stream;
        static std::string log_path;
        static bool flag_save_to_file;
    };

    class Logger
    {
        friend class LogHelper;

    public:
        static void set_log_path(const std::string& path);
        static void start();
        static void stop();

    private:
        static void log_item(const std::string& s);
        static void process_event(const std::string& s);
        static void thread_entry(void* param);
        static std::string m_log_path;
        static std::deque<std::string> m_queue;
    };

    extern std::ostream& get_nil_stream();

    void default_logger_handler_func(const std::string& s);

#define LOG_ERR                                                                                    \
    nnfusion::LogHelper(nnfusion::LOG_TYPE::_LOG_TYPE_ERROR,                                       \
                        nnfusion::trim_file_name(PROJECT_ROOT_DIR, __FILE__),                      \
                        __LINE__,                                                                  \
                        nnfusion::default_logger_handler_func)                                     \
        .stream()

#define LOG_WARN                                                                                   \
    nnfusion::LogHelper(nnfusion::LOG_TYPE::_LOG_TYPE_WARNING,                                     \
                        nnfusion::trim_file_name(PROJECT_ROOT_DIR, __FILE__),                      \
                        __LINE__,                                                                  \
                        nnfusion::default_logger_handler_func)                                     \
        .stream()

#define LOG_INFO                                                                                   \
    nnfusion::LogHelper(nnfusion::LOG_TYPE::_LOG_TYPE_INFO,                                        \
                        nnfusion::trim_file_name(PROJECT_ROOT_DIR, __FILE__),                      \
                        __LINE__,                                                                  \
                        nnfusion::default_logger_handler_func)                                     \
        .stream()

#ifdef NGRAPH_DEBUG_ENABLE
#define NGRAPH_DEBUG                                                                               \
    nnfusion::LogHelper(nnfusion::LOG_TYPE::_LOG_TYPE_DEBUG,                                       \
                        nnfusion::trim_file_name(PROJECT_ROOT_DIR, __FILE__),                      \
                        __LINE__,                                                                  \
                        nnfusion::default_logger_handler_func)                                     \
        .stream()
#else
#define NGRAPH_DEBUG nnfusion::get_nil_stream()
#endif
}
