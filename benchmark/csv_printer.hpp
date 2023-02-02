// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CSV_PRINTER20220221_H
#define CSV_PRINTER20220221_H

#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <string>
#include <vector>

class csv_printer {
  public:
    csv_printer(std::ostream *out, std::initializer_list<std::string> names)
        : out_(out), col_(0), num_cols_(names.size()) {
        for (auto const &name : names) {
            *this << name;
        }
    }
    csv_printer(std::ostream *out, std::vector<std::string> const &names)
        : out_(out), col_(0), num_cols_(names.size()) {
        for (auto const &name : names) {
            *this << name;
        }
    }
    csv_printer(csv_printer &&other) = default;
    csv_printer(csv_printer const &other) = default;
    csv_printer &operator=(csv_printer &&other) = default;
    csv_printer &operator=(csv_printer const &other) = default;

    template <typename T> csv_printer &operator<<(T const &item) {
        *out_ << item;
        if (++col_ >= num_cols_) {
            *out_ << std::endl;
            col_ = 0;
        } else {
            *out_ << ",";
        }
        return *this;
    }

  private:
    std::ostream *out_ = nullptr;
    std::size_t col_;
    std::size_t num_cols_;
};

#endif // CSV_PRINTER20220221_H
