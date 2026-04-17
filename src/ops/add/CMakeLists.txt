target_sources(${LIB_TARGET} PRIVATE
  add_cpu.cpp
)

if(MY_KERNELS_WITH_CUDA)
  target_sources(${LIB_TARGET} PRIVATE
    add_cuda.cu
  )
endif()
