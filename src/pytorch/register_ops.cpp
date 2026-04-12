#include "my_kernels/add.h"

#include <torch/library.h>
#include <Python.h>

TORCH_LIBRARY(my_kernels, m) {
    m.def("add(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_kernels, CPU, m) {
    m.impl("add", &add_cpu);
}

#ifdef WITH_CUDA
TORCH_LIBRARY_IMPL(my_kernels, CUDA, m) {
    m.impl("add", &add_cuda);
}
#endif

extern "C" PyObject* PyInit__C(void) {
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr,
    };
    return PyModule_Create(&module_def);
}
