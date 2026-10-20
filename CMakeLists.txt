# cuokada: elastic dislocations in a half space, on the cpu and the gpu
#
# Copyright (c) 2026 California Institute of Technology, MIT License; see LICENSE

cmake_minimum_required(VERSION 3.24)
project(cuokada VERSION 0.1.0 LANGUAGES CXX)

option(CUOKADA_CUDA "build the cuda backend" ON)
option(CUOKADA_FAST_MATH "fast math in the single precision cuda kernels" ON)
option(CUOKADA_PYTHON "build the python bindings" ON)
option(CUOKADA_TESTS "build the c and fortran tests" ON)
option(CUOKADA_FORTRAN "build the fortran module, if there is a compiler" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

include(GNUInstallDirs)
find_package(OpenMP)

# cuda if asked for and found
if(CUOKADA_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
  else()
    message(STATUS "cuokada: no cuda compiler, building the cpu backend only")
    set(CUOKADA_CUDA OFF)
  endif()
endif()

# the sources, shared by the library and the python module
add_library(cuokada_objects OBJECT src/cpu.cc)
target_include_directories(cuokada_objects PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
if(OpenMP_CXX_FOUND)
  target_link_libraries(cuokada_objects PUBLIC OpenMP::OpenMP_CXX)
endif()
if(CUOKADA_CUDA)
  target_sources(cuokada_objects PRIVATE src/cuda.cu)
  target_compile_definitions(cuokada_objects PUBLIC CUOKADA_HAVE_CUDA)
  # the runtime of the toolkit that compiles, not one elsewhere on the prefix path
  find_library(CUOKADA_CUDART cudart_static PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} NO_DEFAULT_PATH REQUIRED)
  find_package(Threads REQUIRED)
  target_link_libraries(cuokada_objects PUBLIC ${CUOKADA_CUDART} Threads::Threads ${CMAKE_DL_LIBS})
  if(UNIX AND NOT APPLE)
    target_link_libraries(cuokada_objects PUBLIC rt)
  endif()
  if(CUOKADA_FAST_MATH)
    set_source_files_properties(src/cuda.cu PROPERTIES COMPILE_OPTIONS --use_fast_math)
  endif()
endif()

# the shared library, for c, c++, fortran and cuda codes
add_library(cuokada SHARED)
target_link_libraries(cuokada PRIVATE cuokada_objects)
target_include_directories(cuokada PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
# the library, headers and fortran module go under the install prefix, or, from pip, into the lib
# and include of the environment the wheel is installed into, through its data directory; a wheel
# can't hold the symlinks of a versioned library, so there the library is unversioned
if(SKBUILD)
  set(CUOKADA_LIBDIR ${SKBUILD_DATA_DIR}/lib)
  set(CUOKADA_INCLUDEDIR ${SKBUILD_DATA_DIR}/include)
else()
  set(CUOKADA_LIBDIR ${CMAKE_INSTALL_LIBDIR})
  set(CUOKADA_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})
  set_target_properties(cuokada PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
endif()
install(TARGETS cuokada LIBRARY DESTINATION ${CUOKADA_LIBDIR})
install(DIRECTORY include/cuokada DESTINATION ${CUOKADA_INCLUDEDIR})

# the python module
if(CUOKADA_PYTHON)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_cuokada python/bindings.cc)
  target_link_libraries(_cuokada PRIVATE cuokada_objects)
  # skbuild installs into the wheel; a plain cmake build into the site packages of {Python}
  if(SKBUILD)
    set(CUOKADA_PYTHON_DIR cuokada)
  else()
    set(CUOKADA_PYTHON_DIR ${Python_SITEARCH}/cuokada)
  endif()
  install(TARGETS _cuokada LIBRARY DESTINATION ${CUOKADA_PYTHON_DIR})
  install(FILES python/cuokada/__init__.py DESTINATION ${CUOKADA_PYTHON_DIR})
  # a copy of the package next to the module, for the tests in the build directory
  configure_file(python/cuokada/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/cuokada/__init__.py COPYONLY)
  set_target_properties(_cuokada PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cuokada)
endif()

# the fortran module
if(CUOKADA_FORTRAN)
  include(CheckLanguage)
  check_language(Fortran)
  if(CMAKE_Fortran_COMPILER)
    enable_language(Fortran)
    add_library(cuokada_fortran STATIC fortran/cuokada.f90)
    set_target_properties(cuokada_fortran PROPERTIES Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/fortran)
    target_include_directories(cuokada_fortran PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/fortran)
    target_link_libraries(cuokada_fortran PUBLIC cuokada)
    install(FILES fortran/cuokada.f90 ${CMAKE_CURRENT_BINARY_DIR}/fortran/cuokada.mod
      DESTINATION ${CUOKADA_INCLUDEDIR}/cuokada)
  else()
    set(CUOKADA_FORTRAN OFF)
  endif()
else()
  set(CUOKADA_FORTRAN OFF)
endif()

# the tests
if(CUOKADA_TESTS)
  enable_language(C)
  enable_testing()
  add_executable(test_capi tests/test_capi.c)
  target_link_libraries(test_capi PRIVATE cuokada m)
  add_test(NAME capi COMMAND test_capi)
  if(CUOKADA_FORTRAN)
    add_executable(test_fortran tests/test_fortran.f90)
    target_link_libraries(test_fortran PRIVATE cuokada_fortran)
    add_test(NAME fortran COMMAND test_fortran)
  endif()
  if(CUOKADA_PYTHON)
    foreach(t rect tri cuda interop)
      add_test(NAME python_${t} COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_${t}.py)
      set_tests_properties(python_${t} PROPERTIES ENVIRONMENT CUOKADA_BUILD=${CMAKE_CURRENT_BINARY_DIR})
    endforeach()
  endif()
endif()

# end of file
