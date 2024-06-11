#ifndef TEST_SIGNAL_20240610_HPP
#define TEST_SIGNAL_20240610_HPP

#include "bbfft/configuration.hpp"

#include <iosfwd>

void test_signal_1d(void *x, bbfft::configuration const &cfg, long first_mode);
bool check_signal_1d(void *x, bbfft::configuration const &cfg, long first_mode, std::ostream *os);

#endif // TEST_SIGNAL_20240610_HPP
