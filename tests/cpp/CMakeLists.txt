set(MY_KERNELS_TEST_SUPPORT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_add_cpu
  add/test_add_cpu.cpp
)

target_include_directories(test_add_cpu PRIVATE
  ${MY_KERNELS_TEST_SUPPORT_DIR}
)

target_link_libraries(test_add_cpu PRIVATE
  my_kernels::my_kernels
)

add_test(NAME cpp.add.cpu COMMAND test_add_cpu)

if(MY_KERNELS_WITH_CUDA)
  add_executable(test_add_cuda
    add/test_add_cuda.cpp
  )

  target_include_directories(test_add_cuda PRIVATE
    ${MY_KERNELS_TEST_SUPPORT_DIR}
  )

  target_link_libraries(test_add_cuda PRIVATE
    my_kernels::my_kernels
  )

  add_test(NAME cpp.add.cuda COMMAND test_add_cuda)
endif()
