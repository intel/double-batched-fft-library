# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED COMPONENTS breathe sphinx_book_theme)

set(DOXYFILE_IN Doxyfile.in)
set(DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(DOXYFILE_INPUT
    "${INCLUDE_DIR}/bbfft ${INCLUDE_DIR}/bbfft/cl ${INCLUDE_DIR}/bbfft/sycl ${INCLUDE_DIR}/bbfft/ze ${INCLUDE_DIR}/clir")
set(DOXY_XML_DIR ${CMAKE_CURRENT_BINARY_DIR}/xml)
set(DOXY_XML_INDEX ${DOXY_XML_DIR}/index.xml)
set(SPHINX_SRC ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/_build)
set(SPHINX_INDEX ${SPHINX_BUILD}/index.html)

configure_file(${DOXYFILE_IN} ${DOXYFILE} @ONLY)

file(GLOB_RECURSE PUBLIC_HEADER_FILES "${CMAKE_SOURCE_DIR}/include/*.hpp")
add_custom_command(OUTPUT ${DOXY_XML_INDEX}
    MAIN_DEPENDENCY ${DOXYFILE_IN} ${DOXYFILE}
    DEPENDS ${PUBLIC_HEADER_FILES}
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(doc-doxygen ALL DEPENDS ${DOXY_XML_INDEX})

set(RST_DOC_FILES
    index.rst
    api/configuration.rst
    api/generator.rst
    api/index.rst
    api/plan.rst
    api/utilities.rst
    manual/build.rst
    manual/data-layout.rst
    manual/examples.rst
    manual/index.rst
    manual/plans.rst
)
add_custom_command(OUTPUT ${SPHINX_INDEX}
    MAIN_DEPENDENCY ${SPHINX_SRC}/conf.py
    DEPENDS ${RST_DOC_FILES} ${DOXY_XML_INDEX}
    COMMAND ${Sphinx_BUILD_COMMAND}
        ${SPHINX_SRC} ${SPHINX_BUILD}
        -Dbreathe_projects.api=${DOXY_XML_DIR}
    COMMAND_EXPAND_LISTS
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
add_custom_target(docs ALL DEPENDS ${SPHINX_INDEX})

include(GNUInstallDirs)
install(DIRECTORY ${SPHINX_BUILD}/ DESTINATION ${CMAKE_INSTALL_DOCDIR})
