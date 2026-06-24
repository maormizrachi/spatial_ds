#ifndef SPATIAL_DS_ERROR_HPP
#define SPATIAL_DS_ERROR_HPP

#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <utility>

class SpatialDsError : public std::runtime_error
{
public:
    explicit SpatialDsError(const std::string &msg)
        : std::runtime_error(msg), msg_(msg) {}

    template<typename T>
    void addEntry(const std::string &name, const T &value)
    {
        std::ostringstream oss;
        oss << value;
        entries_.emplace_back(name, oss.str());
        rebuildWhat();
    }

    const std::string &getErrorMessage() const { return msg_; }

    const std::vector<std::pair<std::string, std::string>> &getEntries() const { return entries_; }

private:
    void rebuildWhat()
    {
        std::string full = msg_;
        for(const auto &[name, val] : entries_)
        {
            full += "\n  " + name + ": " + val;
        }
        full_ = std::move(full);
    }

    const char *what() const noexcept override { return full_.empty() ? msg_.c_str() : full_.c_str(); }

    std::string msg_;
    std::string full_;
    std::vector<std::pair<std::string, std::string>> entries_;
};

#endif // SPATIAL_DS_ERROR_HPP
