cmake_minimum_required(VERSION 3.18)

project(my_kernels
  VERSION 0.1.0
  DESCRIPTION "A collection of CUDA kernels for learning purpose."
  LANGUAGES CXX
)

option(BUILD_CUDA "Build CUDA kernels if a CUDA compiler is available." ON)

find_package(Torch REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# PyTorch's CMake docs still show applying TORCH_CXX_FLAGS after find_package(Torch).
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include(CheckLanguage)

set(MY_KERNELS_SOURCES
  src/pytorch/register_ops.cpp
  src/ops/add/add_cpu.cpp
)

if(BUILD_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    list(APPEND MY_KERNELS_SOURCES
      src/ops/add/add_cuda.cu
    )
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}. Building CUDA kernels.")
  else()
    message(WARNING "CUDA compiler not found. CUDA kernels will not be built.")
  endif()
endif()

Python3_add_library(${PROJECT_NAME} MODULE WITH_SOABI ${MY_KERNELS_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${Python3_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  ${TORCH_LIBRARIES}
)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

if(CMAKE_CUDA_COMPILER)
  target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_CUDA)
  set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
  )
endif()
