#ifndef CL_MEM_20240610_HPP
#define CL_MEM_20240610_HPP

#include "bbfft/mem.hpp"

#include <CL/cl.h>

namespace bbfft {

/**
 * @brief Auto mem type specialization for cl_mem buffers
 *
 * @tparam T type
 */
template <> struct auto_mem_type<cl_mem> {
    //! Memory type value
    constexpr static mem_type value = mem_type::buffer;
};

} // namespace bbfft

#endif // CL_MEM_20240610_HPP
