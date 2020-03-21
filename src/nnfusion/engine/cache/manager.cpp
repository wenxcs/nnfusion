// Microsoft (c) 2019, NNFusion Team

#include "./manager.hpp"

using namespace nnfusion::cache;

sqlite3* KernelCacheManager::kernel_cache = NULL;

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
   tag        TEXT NOT NULL,
   function   TEXT NOT NULL,
   PRIMARY KEY (identifier, tag));
)";
            NNFUSION_CHECK(SQLITE_OK == sqlite3_exec(kernel_cache, table_create, NULL, 0, NULL));
            valid = true;
        }
        else
        {
            valid = false;
            NNFUSION_LOG(NNFUSION_WARNING) << "Invalid path to kernel cache: " << m_path;
        }
    }
}

KernelCacheManager::~KernelCacheManager()
{
    sqlite3_close(kernel_cache);
}

std::string KernelCacheManager::fetch(std::string identifier, std::string tag)
{
    NNFUSION_LOG(DEBUG) << "Trying to fetch kernel " << identifier << " with tag: " << tag;
    sqlite3_stmt* pStmt;
    const char* fetch = R"(
SELECT function FROM KernelCache WHERE (identifier = ?) AND (tag = ?);
)";
    NNFUSION_CHECK(SQLITE_OK == sqlite3_prepare(kernel_cache, fetch, -1, &pStmt, 0));
    sqlite3_bind_text(pStmt, 1, identifier.data(), identifier.size(), SQLITE_STATIC);
    sqlite3_bind_text(pStmt, 2, tag.data(), tag.size(), SQLITE_STATIC);

    if (SQLITE_DONE != sqlite3_step(pStmt))
    {
        std::string function = std::string((char*)sqlite3_column_text(pStmt, 0));
        NNFUSION_CHECK(SQLITE_DONE == sqlite3_step(pStmt));
        NNFUSION_CHECK(SQLITE_OK == sqlite3_finalize(pStmt));
        NNFUSION_LOG(INFO) << "Using cached kernel " << identifier << "with tag: " << tag;
        return function;
    }
    else
    {
        NNFUSION_LOG(DEBUG) << "Failed to fetch, fallback plan will be used";
        return std::string("");
    }
}