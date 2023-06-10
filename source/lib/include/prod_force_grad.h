#pragma once

namespace deepmd {

template <typename FPTYPE>
void prod_force_grad_a_cpu(FPTYPE* grad_net,
                           const FPTYPE* grad,
                           const FPTYPE* env_deriv,
                           const int* nlist,
                           const int nloc,
                           const int nnei,
                           const int nframes);

template <typename FPTYPE>
void prod_force_grad_r_cpu(FPTYPE* grad_net,
                           const FPTYPE* grad,
                           const FPTYPE* env_deriv,
                           const int* nlist,
                           const int nloc,
                           const int nnei,
                           const int nframes);

#if GOOGLE_CUDA
template <typename FPTYPE>
void prod_force_grad_a_gpu_cuda(FPTYPE* grad_net,
                                const FPTYPE* grad,
                                const FPTYPE* env_deriv,
                                const int* nlist,
                                const int nloc,
                                const int nnei,
                                const int nframes);

template <typename FPTYPE>
void prod_force_grad_r_gpu_cuda(FPTYPE* grad_net,
                                const FPTYPE* grad,
                                const FPTYPE* env_deriv,
                                const int* nlist,
                                const int nloc,
                                const int nnei,
                                const int nframes);
#endif  // GOOGLE_CUDA

#if TENSORFLOW_USE_ROCM
template <typename FPTYPE>
void prod_force_grad_a_gpu_rocm(FPTYPE* grad_net,
                                const FPTYPE* grad,
                                const FPTYPE* env_deriv,
                                const int* nlist,
                                const int nloc,
                                const int nnei,
                                const int nframes);

template <typename FPTYPE>
void prod_force_grad_r_gpu_rocm(FPTYPE* grad_net,
                                const FPTYPE* grad,
                                const FPTYPE* env_deriv,
                                const int* nlist,
                                const int nloc,
                                const int nnei,
                                const int nframes);
#endif  // TENSORFLOW_USE_ROCM
}  // namespace deepmd
