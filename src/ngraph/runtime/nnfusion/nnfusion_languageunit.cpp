// Microsoft (c) 2019, Wenxiang Hu
#include "ngraph/runtime/nnfusion/nnfusion_languageunit.hpp"

LanguageUnit::LanguageUnit(const string symbol)
    : symbol(symbol)
{
}

LanguageUnit::LanguageUnit(const string symbol, const string code)
    : symbol(symbol)
{
    (*this) << code;
}

bool LanguageUnit::change_symbol(const string symbol)
{
    this->symbol = symbol;
    return true;
}

bool LanguageUnit::require(const string required)
{
    this->required.insert(required);
    return true;
}

bool LanguageUnit::require(shared_ptr<LanguageUnit> lu)
{
    this->required.insert(lu->get_symbol());
    this->local_symbol.emplace(lu->get_symbol(), lu);
    return true;
}

string LanguageUnit::collect_code()
{
    LanguageUnit lu;
    for (auto& it : this->required)
    {
        assert_bool(this->local_symbol.find(it) != this->local_symbol.end())
            << "Cannot collect code from non-existed Language Unint.";
        assert_nullptr(this->local_symbol[it])
            << "Cannot collect code from non-existed null pointer.";
        lu << this->local_symbol[it]->collect_code() << "\n";
    }
    lu << "// symbol: " << this->symbol << "\n";
    auto str = this->get_code();
    if (str.empty())
        lu << "// Empty Code\n";
    else
        lu << str;
    return lu.get_code();
}