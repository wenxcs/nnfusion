// Microsoft (c) 2019, Wenxiang Hu
// This file is modified from nnfusion Log

#include <chrono>
#include <condition_variable>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>

#include "log.hpp"

using namespace std;
using namespace nnfusion;

bool nnfusion::LogHelper::flag_save_to_file = false;
std::string nnfusion::LogHelper::log_path = "";

ostream& nnfusion::get_nil_stream()
{
    static stringstream nil;
    return nil;
}

void nnfusion::default_logger_handler_func(const string& s)
{
    cout << s << endl;
}

LogHelper::LogHelper(LOG_TYPE type,
                     const char* file,
                     int line,
                     function<void(const string&)> handler_func)
    : m_handler_func(handler_func)
{
    switch (type)
    {
    case LOG_TYPE::_LOG_TYPE_ERROR: m_stream << "[ERR] "; break;
    case LOG_TYPE::_LOG_TYPE_WARNING: m_stream << "[WARN] "; break;
    case LOG_TYPE::_LOG_TYPE_INFO: m_stream << "[INFO] "; break;
    case LOG_TYPE::_LOG_TYPE_DEBUG: m_stream << "[DEBUG] "; break;
    }

    time_t tt = chrono::system_clock::to_time_t(chrono::system_clock::now());
    auto tm = gmtime(&tt);
    char buffer[256];
    strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%Sz", tm);
    m_stream << buffer << " ";

    m_stream << file;
    m_stream << " " << line;
    m_stream << "\t";

    //Todo(wenxh): Potential Thread Blocking if writting to file
    if (flag_save_to_file)
    {
        std::ofstream log_file(log_path, std::ios::out | std::ios::app);
        log_file << m_stream.str() << endl;
    }
}

LogHelper::~LogHelper()
{
    if (m_handler_func)
    {
        m_handler_func(m_stream.str());
    }
    // Logger::log_item(m_stream.str());
}
