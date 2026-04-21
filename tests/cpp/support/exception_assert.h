#pragma once

#include <gtest/gtest.h>

#include <functional>
#include <stdexcept>
#include <string_view>

namespace my_kernels::test {

template <typename Fn>
void expect_throws_with(Fn&& fn, std::string_view expected_substring) {
    try {
        fn();
    } catch (const std::exception& ex) {
        EXPECT_NE(std::string_view(ex.what()).find(expected_substring), std::string_view::npos)
            << "expected exception containing \"" << expected_substring << "\", got: "
            << ex.what();
        return;
    }

    ADD_FAILURE() << "expected exception containing \"" << expected_substring << '"';
}

} // namespace my_kernels::test
