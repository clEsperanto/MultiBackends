#include "backend.hpp"
#include "cle_preamble_cu.h"

namespace cle
{

CUDABackend::CUDABackend()
{
#if CLE_CUDA
  cuInit(0);
#endif
}

auto
CUDABackend::getDevices(const std::string & type) const -> std::vector<Device::Pointer>
{
#if CLE_CUDA
  int  deviceCount;
  auto error = cuDeviceGetCount(&deviceCount);
  if (error != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to get CUDA device count (" + std::to_string(error) + ").");
  }
  std::vector<Device::Pointer> devices;
  for (int i = 0; i < deviceCount; i++)
  {
    devices.push_back(std::make_shared<CUDADevice>(i));
  }
  return devices;
#else
  throw std::runtime_error("Error: CUDA is not enabled");
#endif
}

auto
CUDABackend::getDevice(const std::string & name, const std::string & type) const -> Device::Pointer
{
#if CLE_CUDA
  auto devices = getDevices(type);
  auto ite = std::find_if(devices.begin(), devices.end(), [&name](const Device::Pointer & dev) {
    return dev->getName().find(name) != std::string::npos;
  });
  if (ite != devices.end())
  {
    return std::move(*ite);
  }
  if (!devices.empty())
  {
    std::cerr << "WARNING: Device with name '" << name << "' not found. Using default device." << std::endl;
    return std::move(devices.back());
  }
  return nullptr;
#else
  throw std::runtime_error("Error: CUDA is not enabled");
#endif
}

auto
CUDABackend::getDevicesList(const std::string & type) const -> std::vector<std::string>
{
#if CLE_CUDA
  auto                     devices = getDevices(type);
  std::vector<std::string> deviceList;
  for (int i = 0; i < devices.size(); i++)
  {
    deviceList.emplace_back(devices[i]->getName());
  }
  return deviceList;
#else
  throw std::runtime_error("Error: CUDA is not enabled");
#endif
}

auto
CUDABackend::getType() const -> Backend::Type
{
  return Backend::Type::CUDA;
}

auto
CUDABackend::allocateMemory(const Device::Pointer & device, const size_t & size, void ** data_ptr) const -> void
{
#if CLE_CUDA
  auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
  auto err = cuCtxSetCurrent(cuda_device->getCUDAContext());
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to get context from device (" + std::to_string(err) + ").");
  }
  CUdeviceptr mem;
  err = cuMemAlloc(&mem, size);
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to allocate memory (buffer) with error code " + std::to_string(err));
  }
  *data_ptr = reinterpret_cast<void *>(mem);
#else
  throw std::runtime_error("Error: CUDA is not enabled");
#endif
}

auto
CUDABackend::allocateMemory(const Device::Pointer & device,
                            const size_t &          width,
                            const size_t &          height,
                            const size_t &          depth,
                            const dType &           dtype,
                            void **                 data_ptr) const -> void
{
#if CLE_CUDA
  auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
  auto err = cuCtxSetCurrent(cuda_device->getCUDAContext());
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to get context from device (" + std::to_string(err) + ").");
  }
  CUarray        array;
  CUarray_format format;
  switch (dtype)
  {
    case dType::Float:
      format = CU_AD_FORMAT_FLOAT;
      break;
    case dType::Int8:
      format = CU_AD_FORMAT_SIGNED_INT8;
      break;
    case dType::UInt8:
      format = CU_AD_FORMAT_UNSIGNED_INT8;
      break;
    case dType::Int16:
      format = CU_AD_FORMAT_SIGNED_INT16;
      break;
    case dType::UInt16:
      format = CU_AD_FORMAT_UNSIGNED_INT16;
      break;
    case dType::Int32:
      format = CU_AD_FORMAT_SIGNED_INT32;
      break;
    case dType::UInt32:
      format = CU_AD_FORMAT_UNSIGNED_INT32;
      break;
    default:
      format = CU_AD_FORMAT_FLOAT;
      std::cerr << "Warning: Unsupported data type for 'image', default type 'float' will be used." << std::endl;
      break;
  }
  if (depth > 1)
  {
    CUDA_ARRAY3D_DESCRIPTOR desc;
    desc.Width = width;
    desc.Height = height;
    desc.Depth = depth;
    desc.Format = format;
    desc.NumChannels = 1;
    desc.Flags = 0;
    err = cuArray3DCreate(&array, &desc);
  }
  else
  {
    CUDA_ARRAY_DESCRIPTOR desc;
    desc.Width = width;
    desc.Height = height;
    desc.Format = format;
    desc.NumChannels = 1;
    err = cuArrayCreate(&array, &desc);
  }
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to allocate memory (image) with error code " + std::to_string(err));
  }
  *data_ptr = reinterpret_cast<void *>(array);
#else
  throw std::runtime_error("Error: CUDA is not enabled");
#endif
}

auto
CUDABackend::freeMemory(const Device::Pointer & device, const mType & mtype, void ** data_ptr) const -> void
{
#if CLE_CUDA
  auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
  auto err = cuCtxSetCurrent(cuda_device->getCUDAContext());
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to get context from device (" + std::to_string(err) + ").");
  }
  if (mtype == mType::Image)
  {
    err = cuArrayDestroy(reinterpret_cast<CUarray>(*data_ptr));
  }
  else
  {
    err = cuMemFree(reinterpret_cast<CUdeviceptr>(*data_ptr));
  }
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to free memory with error code " + std::to_string(err) + ".");
  }
#else
  throw std::runtime_error("Error: CUDA is not enabled");
#endif
}

auto
CUDABackend::writeMemory(const Device::Pointer & device,
                         void **                 data_ptr,
                         const size_t &          size,
                         const void *            host_ptr) const -> void
{
#if CLE_CUDA
  auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
  auto err = cuCtxSetCurrent(cuda_device->getCUDAContext());
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to get context from device (" + std::to_string(err) + ").");
  }
  err = cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(*data_ptr), host_ptr, size);
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to write on device (host -> buffer) with error code " +
                             std::to_string(err));
  }
#else
  throw std::runtime_error("Error: CUDA is not enabled");
#endif
}

auto
CUDABackend::writeMemory(const Device::Pointer & device,
                         void **                 data_ptr,
                         const size_t &          width,
                         const size_t &          height,
                         const size_t &          depth,
                         const size_t &          bytes,
                         const void *            host_ptr) const -> void
{
#if CLE_CUDA
  auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
  auto err = cuCtxSetCurrent(cuda_device->getCUDAContext());
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to get context from device (" + std::to_string(err) + ").");
  }
  if (depth > 1)
  {
    CUDA_MEMCPY3D copyParams = { 0 };
    copyParams.srcMemoryType = CU_MEMORYTYPE_HOST;
    copyParams.srcHost = host_ptr;
    copyParams.srcPitch = width * bytes;
    copyParams.srcHeight = height;
    copyParams.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyParams.dstArray = reinterpret_cast<CUarray>(*data_ptr);
    copyParams.dstPitch = width * bytes;
    copyParams.dstHeight = height;
    copyParams.WidthInBytes = width * bytes;
    copyParams.Height = height;
    copyParams.Depth = depth;
    err = cuMemcpy3D(&copyParams);
  }
  else
  {
    CUDA_MEMCPY2D copyParams = { 0 };
    copyParams.srcMemoryType = CU_MEMORYTYPE_HOST;
    copyParams.srcHost = host_ptr;
    copyParams.srcPitch = width * bytes;
    copyParams.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyParams.dstArray = reinterpret_cast<CUarray>(*data_ptr);
    copyParams.dstPitch = width * bytes;
    copyParams.WidthInBytes = width * bytes;
    copyParams.Height = height;
    err = cuMemcpy2D(&copyParams);
  }
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to write on device (host -> image) with error code " +
                             std::to_string(err));
  }
#else
  throw std::runtime_error("Error: CUDA is not enabled");
#endif
}

auto
CUDABackend::readMemory(const Device::Pointer & device,
                        const void **           data_ptr,
                        const size_t &          size,
                        void *                  host_ptr) const -> void
{
#if CLE_CUDA
  auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
  auto err = cuCtxSetCurrent(cuda_device->getCUDAContext());
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to get context from device (" + std::to_string(err) + ").");
  }
  err = cuMemcpyDtoH(host_ptr, reinterpret_cast<CUdeviceptr>(*data_ptr), size);
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to read memory (buffer -> host) with error code " +
                             std::to_string(err));
  }
#else
  throw std::runtime_error("Error: CUDA is not enabled");
#endif
}

auto
CUDABackend::readMemory(const Device::Pointer & device,
                        const void **           data_ptr,
                        const size_t &          width,
                        const size_t &          height,
                        const size_t &          depth,
                        const size_t &          bytes,
                        void *                  host_ptr) const -> void
{
#if CLE_CUDA
  auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
  auto err = cuCtxSetCurrent(cuda_device->getCUDAContext());
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to get context from device (" + std::to_string(err) + ").");
  }

  if (depth > 1)
  {
    CUDA_MEMCPY3D copyParams = { 0 };
    copyParams.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copyParams.srcArray = reinterpret_cast<CUarray>(const_cast<void *>(*data_ptr));
    copyParams.srcPitch = width * bytes;
    copyParams.srcHeight = height;
    copyParams.dstMemoryType = CU_MEMORYTYPE_HOST;
    copyParams.dstHost = host_ptr;
    copyParams.dstPitch = width * bytes;
    copyParams.dstHeight = height;
    copyParams.WidthInBytes = width * bytes;
    copyParams.Height = height;
    copyParams.Depth = depth;
    err = cuMemcpy3D(&copyParams);
  }
  else
  {
    CUDA_MEMCPY2D copyParams = { 0 };
    copyParams.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copyParams.srcArray = reinterpret_cast<CUarray>(const_cast<void *>(*data_ptr));
    copyParams.srcPitch = width * bytes;
    copyParams.dstMemoryType = CU_MEMORYTYPE_HOST;
    copyParams.dstHost = host_ptr;
    copyParams.dstPitch = width * bytes;
    copyParams.WidthInBytes = width * bytes;
    copyParams.Height = height;
    err = cuMemcpy2D(&copyParams);
  }
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to read memory (image -> host) with error code " +
                             std::to_string(err));
  }
#else
  throw std::runtime_error("Error: CUDA is not enabled");
#endif
}

auto
CUDABackend::copyMemoryBufferToBuffer(const Device::Pointer & device,
                                      const void **           src_data_ptr,
                                      const size_t &          size,
                                      void **                 dst_data_ptr) const -> void
{
#if CLE_CUDA
  auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
  auto err = cuCtxSetCurrent(cuda_device->getCUDAContext());
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to get context from device (" + std::to_string(err) + ").");
  }
  err = cuMemcpyDtoD(reinterpret_cast<CUdeviceptr>(*dst_data_ptr), reinterpret_cast<CUdeviceptr>(*src_data_ptr), size);
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to copy device memory (buffer -> buffer)  with error code " +
                             std::to_string(err));
  }
#else
  throw std::runtime_error("Error: CUDA is not enabled");
#endif
}

auto
CUDABackend::copyMemoryImageToBuffer(const Device::Pointer & device,
                                     const void **           src_data_ptr,
                                     const size_t &          width,
                                     const size_t &          height,
                                     const size_t &          depth,
                                     const size_t &          bytes,
                                     void **                 dst_data_ptr) const -> void
{
#if CLE_CUDA
  auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
  auto err = cuCtxSetCurrent(cuda_device->getCUDAContext());
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to get context from device (" + std::to_string(err) + ").");
  }
  if (depth > 1)
  {
    CUDA_MEMCPY3D copyParams = { 0 };
    copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copyParams.dstDevice = reinterpret_cast<CUdeviceptr>(*dst_data_ptr);
    copyParams.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copyParams.srcArray = reinterpret_cast<CUarray>(const_cast<void *>(*src_data_ptr));
    copyParams.WidthInBytes = width * bytes;
    copyParams.Height = height;
    copyParams.Depth = depth;
    err = cuMemcpy3D(&copyParams);
  }
  else // if (height > 1)
  {
    CUDA_MEMCPY2D copyParams = { 0 };
    copyParams.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copyParams.dstDevice = reinterpret_cast<CUdeviceptr>(*dst_data_ptr);
    copyParams.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copyParams.srcArray = reinterpret_cast<CUarray>(const_cast<void *>(*src_data_ptr));
    copyParams.WidthInBytes = width * bytes;
    copyParams.Height = height;
    err = cuMemcpy2D(&copyParams);
  }
  // else
  // {
  //   err = cuMemcpyAtoD(reinterpret_cast<CUdeviceptr>(*dst_data_ptr),
  //                         reinterpret_cast<CUarray>(const_cast<void *>(*src_data_ptr)),
  //                         0,
  //                         bytes * width * height * depth);
  // }
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to copy device memory (image -> buffer) with error code " +
                             std::to_string(err));
  }

#else
  throw std::runtime_error("Error: CUDA is not enabled");
#endif
}

auto
CUDABackend::copyMemoryBufferToImage(const Device::Pointer & device,
                                     const void **           src_data_ptr,
                                     const size_t &          width,
                                     const size_t &          height,
                                     const size_t &          depth,
                                     const size_t &          bytes,
                                     void **                 dst_data_ptr) const -> void
{
#if CLE_CUDA
  auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
  auto err = cuCtxSetCurrent(cuda_device->getCUDAContext());
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to get context from device (" + std::to_string(err) + ").");
  }

  if (depth > 1)
  {
    CUDA_MEMCPY3D copyParams = { 0 };
    copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copyParams.srcDevice = reinterpret_cast<const CUdeviceptr>(*src_data_ptr);
    copyParams.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyParams.dstArray = reinterpret_cast<CUarray>(*dst_data_ptr);
    copyParams.WidthInBytes = width * bytes;
    copyParams.Height = height;
    copyParams.Depth = depth;
    err = cuMemcpy3D(&copyParams);
  }
  else // if (height > 1)
  {
    CUDA_MEMCPY2D copyParams = { 0 };
    copyParams.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copyParams.srcDevice = reinterpret_cast<const CUdeviceptr>(*src_data_ptr);
    copyParams.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyParams.dstArray = reinterpret_cast<CUarray>(*dst_data_ptr);
    copyParams.WidthInBytes = width * bytes;
    copyParams.Height = height;
    err = cuMemcpy2D(&copyParams);
  }
  // else
  // {
  //   err = cuMemcpyDtoA(reinterpret_cast<CUarray>(*dst_data_ptr),
  //                         0,
  //                         reinterpret_cast<const CUdeviceptr>(*src_data_ptr),
  //                         bytes * width * height * depth);
  // }
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to copy device memory (buffer -> image) with error code " +
                             std::to_string(err));
  }
#else
  throw std::runtime_error("Error: CUDA is not enabled");
#endif
}

auto
CUDABackend::copyMemoryImageToImage(const Device::Pointer & device,
                                    const void **           src_data_ptr,
                                    const size_t &          width,
                                    const size_t &          height,
                                    const size_t &          depth,
                                    const size_t &          bytes,
                                    void **                 dst_data_ptr) const -> void
{
#if CLE_CUDA
  auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
  auto err = cuCtxSetCurrent(cuda_device->getCUDAContext());
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to get context from device (" + std::to_string(err) + ").");
  }
  if (depth > 1)
  {
    CUDA_MEMCPY3D copyParams = { 0 };
    copyParams.srcArray = reinterpret_cast<CUarray>(const_cast<void *>(*src_data_ptr));
    copyParams.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copyParams.dstArray = reinterpret_cast<CUarray>(*dst_data_ptr);
    copyParams.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyParams.WidthInBytes = width * bytes;
    copyParams.Height = height;
    copyParams.Depth = depth;
    err = cuMemcpy3D(&copyParams);
  }
  else
  {
    CUDA_MEMCPY2D copyParams = { 0 };
    copyParams.srcArray = reinterpret_cast<CUarray>(const_cast<void *>(*src_data_ptr));
    copyParams.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copyParams.dstArray = reinterpret_cast<CUarray>(*dst_data_ptr);
    copyParams.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copyParams.WidthInBytes = width * bytes;
    copyParams.Height = height;
    err = cuMemcpy2D(&copyParams);
  }
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to copy device memory (image -> image) with error code " +
                             std::to_string(err));
  }
#else
  throw std::runtime_error("Error: CUDA is not enabled");
#endif
}

auto
CUDABackend::setMemory(const Device::Pointer & device,
                       void **                 data_ptr,
                       const size_t &          size,
                       const void *            value,
                       const size_t &          value_size) const -> void
{
#if CLE_CUDA
  auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
  auto err = cuCtxSetCurrent(cuda_device->getCUDAContext());
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to get context from device (" + std::to_string(err) + ").");
  }


  // TODO

  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to set memory with error code " + std::to_string(err));
  }
#else
  throw std::runtime_error("Error: CUDA is not enabled");
#endif
}

auto
CUDABackend::setMemory(const Device::Pointer & device,
                       void **                 data_ptr,
                       const size_t &          width,
                       const size_t &          height,
                       const size_t &          depth,
                       const size_t &          bytes,
                       const void *            value) const -> void
{
#if CLE_CUDA
  auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
  auto err = cuCtxSetCurrent(cuda_device->getCUDAContext());
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to get context from device (" + std::to_string(err) + ").");
  }


  // TODO

  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to set memory with error code " + std::to_string(err));
  }
#else
  throw std::runtime_error("Error: CUDA is not enabled");
#endif
}

auto
CUDABackend::loadProgramFromCache(const Device::Pointer & device, const std::string & hash, void * program) const
  -> void
{
#if CLE_CUDA
  auto     cuda_device = std::dynamic_pointer_cast<CUDADevice>(device);
  CUmodule module = nullptr;
  auto     ite = cuda_device->getCache().find(hash);
  if (ite != cuda_device->getCache().end())
  {
    module = ite->second;
  }
  program = module;
#else
  throw std::runtime_error("Error: CUDA is not enabled");
#endif
}

auto
CUDABackend::saveProgramToCache(const Device::Pointer & device, const std::string & hash, void * program) const -> void
{
#if CLE_CUDA
  auto cuda_device = std::dynamic_pointer_cast<CUDADevice>(device);
  cuda_device->getCache().emplace_hint(cuda_device->getCache().end(), hash, reinterpret_cast<CUmodule>(program));
#else
  throw std::runtime_error("Error: CUDA is not enabled");
#endif
}

auto
CUDABackend::buildKernel(const Device::Pointer & device,
                         const std::string &     kernel_source,
                         const std::string &     kernel_name,
                         void *                  kernel) const -> void
{
#if CLE_CUDA
  auto       cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
  CUfunction function = nullptr;

  // TODO

  *(reinterpret_cast<CUfunction *>(kernel)) = function;
#else
  throw std::runtime_error("Error: CUDA is not enabled");
#endif
}

auto
CUDABackend::executeKernel(const Device::Pointer &       device,
                           const std::string &           kernel_source,
                           const std::string &           kernel_name,
                           const std::array<size_t, 3> & global_size,
                           const std::vector<void *> &   args,
                           const std::vector<size_t> &   sizes) const -> void
{
  // TODO
}

auto
CUDABackend::getPreamble() const -> std::string
{
  return kernel::preamble_cu;
}

} // namespace cle
