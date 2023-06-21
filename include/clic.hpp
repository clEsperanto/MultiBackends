#ifndef __INCLUDE_CLIC_HPP
#define __INCLUDE_CLIC_HPP

#if CLE_OPENCL
#  ifndef CL_TARGET_OPENCL_VERSION
#    define CL_TARGET_OPENCL_VERSION 120
#  endif
#  ifdef __APPLE__
#    include <OpenCL/opencl.h>
#  else
#    include <CL/cl.h>
#  endif
#endif

#if CLE_CUDA
#  include <cuda.h>
#  include <cuda_runtime.h>
#  include <cuda_runtime_api.h>
#  include <nvrtc.h>
#endif

#endif // __INCLUDE_CLIC_HPP
