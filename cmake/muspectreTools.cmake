#==============================================================================
# file   muspectreTools.cmake
#
# @author Nicolas Richart <nicolas.,richart@epfl.ch>
#
# @date   11 Jan 2018
#
# @brief  some tool to help to do stuff with cmake in µSpectre
#
# @section LICENSE
#
# Copyright © 2018 Till Junge
#
# µSpectre is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3, or (at
# your option) any later version.
#
# µSpectre is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with µSpectre; see the file COPYING. If not, write to the
# Free Software Foundation, Inc., 59 Temple Place - Suite 330,
# Boston, MA 02111-1307, USA.
#

#[=[.rst:
µSpectreTools
-------------

This module provide some helper functions for µSpectre

::

download_external_project(project_name
  URL <url>
  BACKEND <GIT|HG>
  [TAG <tag>]
  [THIRD_PARTY_SRC_DIR <dir>]
  [NO_UPDATE]
  )

Downloads the external project based on the uri

``THIRD_PARTY_SRC_DIR <dir>``
  Specifies where to download the source

``NO_UPDATE``
  Does not try to update existing download

::

mark_as_advanced_prefix(prefix)

Marks as advanced all variables whoes names start with prefix

::

add_external_dep(package
  [IGNORE_SYSTEM]
  [VERSION <version>]
  [...])

Tries to find the external package and if not found but a local
${package}.cmake files exists, it includes it. The extra arguments are passed
to find_package()

``IGNORE_SYSTEM`` does not do the find_package

``VERSION`` Specifies the required minimum version

]=]
#

if(MUSPECTRETOOLS_INCLUDED)
  return()
endif()
set(MUSPECTRETOOLS_INCLUDED TRUE)

function(download_external_project project_name)
  include(CMakeParseArguments)

  set(_dep_flags
    NO_UPDATE)
  set(_dep_one_variables
    URL
    TAG
    BACKEND
    THIRD_PARTY_SRC_DIR
    )
  set(_dep_multi_variables)
  cmake_parse_arguments(_dep_args
    "${_dep_flags}"
    "${_dep_one_variables}"
    "${_dep_multi_variables}"
    ${ARGN}
    )

  if(NOT _dep_args_URL)
    message(FATAL_ERROR "You have to provide a URL for the project ${project_name}")
  endif()

  if(NOT _dep_args_BACKEND)
    message(FATAL_ERROR "You have to provide a backend to download ${project_name}")
  endif()

  if(_dep_args_TAG)
    set(_ep_tag "${_dep_args_BACKEND}_TAG ${_dep_args_TAG}")
  endif()

  set(_src_dir ${PROJECT_SOURCE_DIR}/third-party/${project_name})
  if (_dep_args_THIRD_PARTY_SRC_DIR)
    set(_src_dir ${_dep_args_THIRD_PARTY_SRC_DIR}/${project_name})
  endif()

  if(EXISTS ${_src_dir}/.DOWNLOAD_SUCCESS AND _dep_args_NO_UPDATE)
    return()
  endif()

  set(_working_dir ${PROJECT_BINARY_DIR}/third-party/${project_name}-download)
  file(WRITE ${_working_dir}/CMakeLists.txt
    "
cmake_minimum_required(VERSION 3.1)
project(${project_name}-download NONE)
include(ExternalProject)
ExternalProject_Add(${project_name}
	SOURCE_DIR ${_src_dir}
	BINARY_DIR ${_working_dir}
	${_dep_args_BACKEND}_REPOSITORY ${_dep_args_URL}
	${_ep_tag}
	CONFIGURE_COMMAND \"\"
	BUILD_COMMAND     \"\"
	INSTALL_COMMAND   \"\"
	TEST_COMMAND      \"\"
	)
")
  message(STATUS "Downloading ${project_name} ${_dep_args_GIT_TAG}")
  execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
    RESULT_VARIABLE _result
    WORKING_DIRECTORY ${_working_dir}
    OUTPUT_FILE ${_working_dir}/configure-out.log
    ERROR_FILE ${_working_dir}/configure-error.log)

  if(_result)
    message(FATAL_ERROR "Something went wrong (${_result}) during the download"
      " process of ${project_name} check the file"
      " ${_working_dir}/configure-error.log for more details:")
    file(STRINGS "${_working_dir}/configure-error.log" ERROR_MSG)
    message("${ERROR_MSG}")
  endif()

  execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    RESULT_VARIABLE _result
    WORKING_DIRECTORY ${_working_dir}
    OUTPUT_FILE ${_working_dir}/build-out.log
    ERROR_FILE ${_working_dir}/build-error.log)

  if(_result)
    message(FATAL_ERROR "Something went wrong (${_result}) during the download"
      " process of ${project_name} check the file"
      " ${_working_dir}/build-error.log for more details")
  endif()

  file(WRITE ${_src_dir}/.DOWNLOAD_SUCCESS "")
endfunction()

# ------------------------------------------------------------------------------
function(mark_as_advanced_prefix prefix)
  get_property(_list DIRECTORY PROPERTY VARIABLES)
  foreach(_var ${_list})
    if("${_var}" MATCHES "^${prefix}")
      mark_as_advanced(${_var})
    endif()
  endforeach()
endfunction()

# ------------------------------------------------------------------------------
function(add_external_package package)
  include(CMakeParseArguments)

  set(_cmake_includes ${PROJECT_SOURCE_DIR}/cmake)
  set(_${package}_external_dir ${PROJECT_BINARY_DIR}/external)

  set(_aep_flags
    IGNORE_SYSTEM
    )
  set(_aep_one_variables
    VERSION
    )
  set(_aep_multi_variables)

  cmake_parse_arguments(_aep_args
    "${_aep_flags}"
    "${_aep_one_variables}"
    "${_aep_multi_variables}"
    ${ARGN}
    )

  if(_aep_args_VERSION)
    set(_${package}_version ${_aep_args_VERSION})
  endif()

  if(NOT EXISTS ${_cmake_includes}/${package}.cmake)
    set(_required REQUIRED)
  endif()

  if(NOT _aep_args_IGNORE_SYSTEM)
    find_package(${package} ${_${package}_version} ${_required} ${_aep_UNPARSED_ARGUMENTS} QUIET)
    if(${package}_FOUND AND NOT ${package}_FOUND_EXTERNAL)
      string(TOUPPER ${package} u_package)
      mark_as_advanced_prefix(${package})
      mark_as_advanced_prefix(${u_package})
      return()
    endif()
  endif()



  if(EXISTS ${_cmake_includes}/${package}.cmake)
    include(${_cmake_includes}/${package}.cmake)
  endif()
  string(TOUPPER ${package} u_package)
  mark_as_advanced_prefix(${package})
  mark_as_advanced_prefix(${u_package})
endfunction()

# ------------------------------------------------------------------------------
function(muSpectre_set_global_compile_options)
  add_compile_options(-Wall -Wextra)

  string(TOLOWER "${CMAKE_BUILD_TYPE}" build_type )
  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    # using Clang
    add_compile_options(-Wno-missing-braces)
    if ("debug" STREQUAL "${build_type}")
      add_compile_options(-O0)
    endif()
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # using GCC
    add_compile_options(-Wno-non-virtual-dtor)
    if (("relwithdebinfo" STREQUAL "${build_type}") OR ("release" STREQUAL "${build_type}" ))
      add_compile_options(-march=native)
    endif()
    if ("debug" STREQUAL "${build_type}" )
      add_compile_options(-O0 -g3 -ggdb3)
    endif()
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    # using Intel C++
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # using Visual Studio C++
  endif()
endfunction()

# ------------------------------------------------------------------------------
