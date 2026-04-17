cmake_minimum_required(VERSION 3.18)

project(my_kernels
  VERSION 0.1.0
  DESCRIPTION "A collection of CUDA kernels for learning purpose."
  LANGUAGES CXX
)

option(BUILD_CUDA "Build CUDA kernels if a CUDA compiler is available." ON)
option(BUILD_PYTHON_EXTENSION "Build Python extension module _C." ON)

set(MY_KERNELS_CMAKE_COMPONENT cpp)
set(MY_KERNELS_PYTHON_COMPONENT python)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(Torch REQUIRED)
if(BUILD_PYTHON_EXTENSION)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
endif()

# PyTorch's CMake docs still show applying TORCH_CXX_FLAGS
# after find_package(Torch).
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include(CheckLanguage)

set(MY_KERNELS_WITH_CUDA OFF)

if(BUILD_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(MY_KERNELS_WITH_CUDA ON)
    message(STATUS
      "CUDA compiler found: ${CMAKE_CUDA_COMPILER}. Building CUDA kernels."
    )
  else()
    message(WARNING "CUDA compiler not found. CUDA kernels will not be built.")
  endif()
endif()

add_subdirectory(src)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/my_kernelsConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/my_kernelsConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/my_kernelsConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/my_kernels"
)

install(
  EXPORT my_kernelsTargets
  FILE my_kernelsTargets.cmake
  NAMESPACE my_kernels::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/my_kernels"
  COMPONENT ${MY_KERNELS_CMAKE_COMPONENT}
)

install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/my_kernelsConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/my_kernelsConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/my_kernels"
  COMPONENT ${MY_KERNELS_CMAKE_COMPONENT}
)
