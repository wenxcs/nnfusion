// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "./manager.hpp"

using namespace nnfusion::cache;

// sqlite3* KernelCacheManager::kernel_cache = NULL;
sqlite3* KernelCacheManager::kernel_cache = nullptr;
KernelCacheManager::KernelCacheManager()
{
    m_path = (getpwuid(getuid())->pw_dir + std::string("/.cache/nnfusion/kernel_cache.db"));

    if (!kernel_cache)
    {
        if (SQLITE_OK == sqlite3_open(m_path.c_str(), &kernel_cache))
        {
            NNFUSION_LOG(INFO) << "Load kernel cache from: " << m_path;
            const char* table_create = R"(
CREATE TABLE IF NOT EXISTS KernelCache(
   identifier TEXT NOT NULL,
   platform   TEXT NOT NULL,
   function   TEXT NOT NULL,
   tags       TEXT DEFAULT "",
   profile    TEXT DEFAULT ""
   );
)";
            NNFUSION_CHECK(SQLITE_OK == sqlite3_exec(kernel_cache, table_create, NULL, 0, NULL));
        }
        else
        {
            NNFUSION_LOG(NNFUSION_WARNING) << "Invalid path to kernel cache: " << m_path;
            kernel_cache = nullptr;
        }
    }
}

KernelCacheManager::~KernelCacheManager()
{
    NNFUSION_CHECK(SQLITE_OK == sqlite3_close(kernel_cache));
    kernel_cache = NULL;
}

std::vector<kernel> KernelCacheManager::fetch_all(std::string identifier, std::string platform)
{
    NNFUSION_LOG(DEBUG) << "Trying to fetch kernel " << identifier << " on platform: " << platform;
    sqlite3_stmt* pStmt;
    const char* fetch = R"(
SELECT function, tags, profile FROM KernelCache WHERE (identifier = ?) AND (platform = ?);
    )";
    NNFUSION_CHECK(SQLITE_OK == sqlite3_prepare(kernel_cache, fetch, -1, &pStmt, 0));
    sqlite3_bind_text(pStmt, 1, identifier.data(), identifier.size(), SQLITE_STATIC);
    sqlite3_bind_text(pStmt, 2, platform.data(), platform.size(), SQLITE_STATIC);

    std::vector<kernel> fetched;
    while (SQLITE_ROW == sqlite3_step(pStmt))
    {
        kernel fetched_kernel;
        fetched_kernel.function = std::string((char*)sqlite3_column_text(pStmt, 0));

        // parse input tags
        size_t pos = 0;
        std::string fetched_tags = std::string((char*)sqlite3_column_text(pStmt, 1));
        while ((pos = fetched_tags.find("_")) != std::string::npos)
        {
            fetched_kernel.tags.insert(fetched_tags.substr(0, pos));
            fetched_tags.erase(0, pos + 1);
        }
        if (fetched_tags != "")
        {
            fetched_kernel.tags.insert(fetched_tags);
        }

        // parse profiling information
        size_t subpos = 0;
        std::string fetched_profile = std::string((char*)sqlite3_column_text(pStmt, 2));
        while ((pos = fetched_profile.find(";")) != std::string::npos)
        {
            subpos = fetched_profile.find(":");
            fetched_kernel.profile[fetched_profile.substr(0, subpos)] =
                stof(fetched_profile.substr(subpos + 1, pos));
            fetched_profile.erase(0, pos + 1);
        }

        fetched.push_back(fetched_kernel);
    }

    NNFUSION_CHECK(SQLITE_OK == sqlite3_finalize(pStmt));
    if (fetched.size() > 0)
    {
        NNFUSION_LOG(INFO) << fetched.size() << " Cached kernel fetched " << identifier
                           << " on: " << platform;
    }
    else
    {
        NNFUSION_LOG(DEBUG) << "Failed to fetch, fallback plan will be uses";
    }
    return fetched;
}

kernel KernelCacheManager::fetch_with_tags(std::string identifier,
                                           std::string platform,
                                           std::set<std::string> tags)
{
    auto fetched = fetch_all(identifier, platform);
    // it's expected that tag set is unique for different implementation
    for (auto matched_kernel : fetched)
    {
        if (matched_kernel.tags == tags)
        {
            return matched_kernel;
        }
    }
    if (!fetched.empty())
    {
        NNFUSION_LOG(DEBUG) << "No kernel with the same tags matched for " << identifier;
    }
    kernel null_kernel;
    null_kernel.function = "";
    return null_kernel;
}