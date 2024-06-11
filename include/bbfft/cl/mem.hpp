#ifndef CL_MEM_20240610_HPP
#define CL_MEM_20240610_HPP

namespace bbfft {

template <> struct auto_mem_type<cl_mem> {
    constexpr static mem_type value = mem_type::buffer;
};

} // namespace bbfft

#endif // CL_MEM_20240610_HPP
