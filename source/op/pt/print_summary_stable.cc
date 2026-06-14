// SPDX-License-Identifier: LGPL-3.0-or-later
// Target the oldest stable ABI version used by this file's helper APIs.
#ifndef TORCH_TARGET_VERSION
#define TORCH_TARGET_VERSION (((0ULL + 2) << 56) | ((0ULL + 10) << 48))
#endif

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

namespace {
namespace th = torch::headeronly;
namespace ts = torch::stable;

ts::Tensor bool_tensor(bool value) {
  return ts::full({1}, value ? 1.0 : 0.0, th::ScalarType::Bool,
                  th::Layout::Strided, ts::Device(th::DeviceType::CPU));
}

ts::Tensor enable_mpi_stable() {
#ifdef USE_MPI
  return bool_tensor(true);
#else
  return bool_tensor(false);
#endif
}
}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(deepmd, m) {
  m.def("enable_mpi() -> Tensor");
  m.impl("enable_mpi", TORCH_BOX(enable_mpi_stable));
}
