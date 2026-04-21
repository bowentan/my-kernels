#pragma once

#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace my_kernels::test {

    inline void require(bool condition, std::string_view expr, std::string_view file, int line) {
        if (condition) {
            return;
        }

        std::ostringstream message;
        message << file << ":" << line << ": requirement failed: " << expr;
        throw std::runtime_error(message.str());
    }

    template <typename Fn>
    inline void require_throws_with(Fn&& fn, std::string_view expected_substring,
                                    std::string_view expr, std::string_view file, int line) {
        try {
            fn();
        } catch (const std::exception& ex) {
            if (std::string_view(ex.what()).find(expected_substring) != std::string_view::npos) {
                return;
            }

            std::ostringstream message;
            message << file << ":" << line << ": expected exception containing \""
                    << expected_substring << "\" from " << expr << ", got: " << ex.what();
            throw std::runtime_error(message.str());
        }

        std::ostringstream message;
        message << file << ":" << line << ": expected exception containing \"" << expected_substring
                << "\" from " << expr;
        throw std::runtime_error(message.str());
    }

    inline int run_test(const std::string& name, const std::function<void()>& fn) {
        try {
            fn();
            std::cout << "[PASS] " << name << '\n';
            return 0;
        } catch (const std::exception& ex) {
            std::cerr << "[FAIL] " << name << ": " << ex.what() << '\n';
            return 1;
        }
    }

} // namespace my_kernels::test

#define REQUIRE(expr) ::my_kernels::test::require((expr), #expr, __FILE__, __LINE__)
#define REQUIRE_THROWS_WITH(expr, expected_substring)                                              \
    ::my_kernels::test::require_throws_with([&] { static_cast<void>(expr); }, expected_substring,  \
                                            #expr, __FILE__, __LINE__)
