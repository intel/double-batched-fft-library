# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(GenerateExportHeader)

set(SOURCES
    builder.cpp
    builtin_function.cpp
    builtin_type.cpp
    data_type.cpp
    expr.cpp
    func.cpp
    op.cpp
    stmt.cpp
    string_util.cpp
    var.cpp
    visitor/codegen_opencl.cpp
    visitor/equal_expr.cpp
    visitor/required_extensions.cpp
    visitor/to_imm.cpp
    visitor/unique_names.cpp
    visitor/unsafe_simplification.cpp
)
set(PUBLIC_HEADERS
    internal/data_type_node.hpp
    internal/function_node.hpp
    internal/macro_helper.hpp
    internal/stmt_node.hpp
    internal/args_impl.hpp
    internal/attr_node.hpp
    internal/expr_node.hpp
    internal/program_node.hpp
    visitor/codegen_opencl.hpp
    visitor/equal_expr.hpp
    visitor/required_extensions.hpp
    visitor/to_imm.hpp
    visitor/unique_names.hpp
    visitor/unsafe_simplification.hpp
    attr.hpp
    attr_defs.hpp
    builder.hpp
    builtin_function.hpp
    builtin_type.hpp
    handle.hpp
    func.hpp
    op.hpp
    prog.hpp
    stmt.hpp
    string_util.hpp
    var.hpp
    virtual_type_list.hpp
    visit.hpp
    data_type.hpp
    expr.hpp
)
list(TRANSFORM PUBLIC_HEADERS PREPEND "${PROJECT_SOURCE_DIR}/include/clir/")

add_library(clir ${SOURCES})
add_library(clir::clir ALIAS clir)
set_target_properties(clir PROPERTIES
                      VERSION ${clir_VERSION}
                      SOVERSION ${clir_VERSION_MAJOR})
target_include_directories(clir PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)
target_compile_features(clir PRIVATE cxx_std_17)
add_common_flags(clir)
set(clir_export_header "${CMAKE_CURRENT_BINARY_DIR}/include/clir/export.hpp")
generate_export_header(clir EXPORT_FILE_NAME "${clir_export_header}")
target_compile_definitions(clir PUBLIC
    "$<$<NOT:$<BOOL:${BUILD_SHARED_LIBS}>>:CLIR_STATIC_DEFINE>")
target_include_directories(clir PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"
)
target_sources(clir PUBLIC FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_BINARY_DIR}/include
    FILES ${PUBLIC_HEADERS} ${clir_export_header})

# installation

if (BUILD_SHARED_LIBS)
    set(type shared)
else ()
    set(type static)
endif ()

set_target_properties(clir PROPERTIES INSTALL_RPATH_USE_LINK_PATH True)
set(CONFIG_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/clir)

install(TARGETS clir EXPORT clir-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILE_SET HEADERS
)
install(EXPORT clir-targets
    FILE clir-${type}-targets.cmake
    NAMESPACE clir::
    DESTINATION ${CONFIG_DESTINATION}
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/clir-config-version.cmake"
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${PROJECT_SOURCE_DIR}/cmake/clir-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/clir-config.cmake"
    INSTALL_DESTINATION ${CONFIG_DESTINATION}
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/clir-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake/clir-config-version.cmake"
    DESTINATION ${CONFIG_DESTINATION}
)
