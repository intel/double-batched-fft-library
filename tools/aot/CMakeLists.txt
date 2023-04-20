# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

include(GNUInstallDirs)

add_executable(bbfft-aot-generate main.cpp args.cpp)
add_executable(bbfft::bbfft-aot-generate ALIAS bbfft-aot-generate)
target_compile_features(bbfft-aot-generate PRIVATE cxx_std_17)
target_link_libraries(bbfft-aot-generate PRIVATE bbfft-base bbfft-level-zero tools-common)

set(CONFIG_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bbfft-aot-generate)

install(TARGETS bbfft-aot-generate EXPORT bbfft-aot-generate-targets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(EXPORT bbfft-aot-generate-targets
    FILE bbfft-aot-generate-targets.cmake
    NAMESPACE "bbfft::"
    DESTINATION ${CONFIG_DESTINATION}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/bbfft-aot-generate-config-version.cmake
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/bbfft-aot-generate-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/bbfft-aot-generate-config.cmake"
    INSTALL_DESTINATION ${CONFIG_DESTINATION}
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/bbfft-aot-generate-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/bbfft-aot-generate-config-version.cmake"
    "${PROJECT_SOURCE_DIR}/cmake/AddAotKernelsToTarget.cmake"
    DESTINATION ${CONFIG_DESTINATION}
)
