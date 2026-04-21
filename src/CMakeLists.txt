set(LIB_TARGET my_kernels)

add_library(${LIB_TARGET})
add_library(my_kernels::my_kernels ALIAS ${LIB_TARGET})
set_target_properties(${LIB_TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${LIB_TARGET}
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(${LIB_TARGET}
  PUBLIC
    ${TORCH_LIBRARIES}
)

target_compile_features(${LIB_TARGET} PUBLIC cxx_std_17)

add_subdirectory(ops)

if(MY_KERNELS_WITH_CUDA)
  target_compile_definitions(${LIB_TARGET} PUBLIC WITH_CUDA)
  set_target_properties(${LIB_TARGET} PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
  )
endif()

install(TARGETS ${LIB_TARGET}
  EXPORT my_kernelsTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT ${MY_KERNELS_CMAKE_COMPONENT}
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  COMPONENT ${MY_KERNELS_CMAKE_COMPONENT}
)

if(BUILD_PYTHON_EXTENSION)
  set(EXT_TARGET _C)
  set(MY_KERNELS_PYTHON_BUILD_ROOT "${PROJECT_BINARY_DIR}/python")
  set(MY_KERNELS_PYTHON_PACKAGE_DIR "${MY_KERNELS_PYTHON_BUILD_ROOT}/my_kernels")

  Python3_add_library(${EXT_TARGET} MODULE WITH_SOABI
    pytorch/register_ops.cpp
  )

  set_target_properties(${EXT_TARGET} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${MY_KERNELS_PYTHON_PACKAGE_DIR}"
  )

  target_include_directories(${EXT_TARGET} PRIVATE
    ${Python3_INCLUDE_DIRS}
  )

  target_link_libraries(${EXT_TARGET} PRIVATE
    ${LIB_TARGET}
  )

  if(MY_KERNELS_WITH_CUDA)
    set_target_properties(${EXT_TARGET} PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
    )
  endif()

  add_custom_command(TARGET ${EXT_TARGET} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${MY_KERNELS_PYTHON_PACKAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${PROJECT_SOURCE_DIR}/python/my_kernels/__init__.py"
      "${MY_KERNELS_PYTHON_PACKAGE_DIR}/__init__.py"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${PROJECT_SOURCE_DIR}/python/my_kernels/_load_lib.py"
      "${MY_KERNELS_PYTHON_PACKAGE_DIR}/_load_lib.py"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${PROJECT_SOURCE_DIR}/python/my_kernels/ops.py"
      "${MY_KERNELS_PYTHON_PACKAGE_DIR}/ops.py"
  )

  install(TARGETS ${EXT_TARGET}
    LIBRARY DESTINATION my_kernels
    COMPONENT ${MY_KERNELS_PYTHON_COMPONENT}
    EXCLUDE_FROM_ALL
  )
endif()
