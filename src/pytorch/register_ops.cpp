#include "my_kernels/vector_add.h"

#include <torch/extension.h>

TORCH_LIBRARY(my_kernels, m) {
    m.def("vector_add(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_kernels, CPU, m) {
    m.impl("vector_add", &vector_add_cpu);
}

#ifdef WITH_CUDA
TORCH_LIBRARY_IMPL(my_kernels, CUDA, m) {
    m.impl("vector_add", &vector_add_cuda);
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
