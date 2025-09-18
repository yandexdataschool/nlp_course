#include <torch/extension.h>

torch::Tensor int4MatmulCUDA(const torch::Tensor &A, const torch::Tensor &B);

torch::Tensor int8MatmulCUDA(const torch::Tensor &A, const torch::Tensor &B);

torch::Tensor int4Matmul(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllContiguous("int4Matmul", {{A, "A", 0}, {B, "B", 1}});
  torch::checkDeviceType("int4Matmul", {A, B}, at::DeviceType::CUDA);
  return int4MatmulCUDA(A, B);
}

torch::Tensor int8Matmul(const torch::Tensor &A, const torch::Tensor &B) {
  torch::checkAllContiguous("int8Matmul", {{A, "A", 0}, {B, "B", 1}});
  torch::checkDeviceType("int8Matmul", {A, B}, at::DeviceType::CUDA);
  return int8MatmulCUDA(A, B);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int4_matmul", &int4Matmul, "int4 matmul (CUDA)");
  m.def("int8_matmul", &int8Matmul, "int8 matmul (CUDA)");
}
