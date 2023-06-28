#include "backend.hpp"
#include "cle_preamble_cu.h"
#include <array>

namespace cle
{

CUDABackend::CUDABackend()
{
#if USE_CUDA
  cuInit(0);
#endif
}

auto
CUDABackend::getDevices(const std::string & type) const -> std::vector<Device::Pointer>
{
#if USE_CUDA
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
#if USE_CUDA
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
#if USE_CUDA
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
#if USE_CUDA
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
#if USE_CUDA
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
#if USE_CUDA
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
#if USE_CUDA
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
#if USE_CUDA
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
#if USE_CUDA
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
#if USE_CUDA
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
#if USE_CUDA
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
#if USE_CUDA
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
#if USE_CUDA
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
#if USE_CUDA
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
                       const float &           value,
                       const dType &           dtype) const -> void
{
#if USE_CUDA
  auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
  auto err = cuCtxSetCurrent(cuda_device->getCUDAContext());
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to get context from device (" + std::to_string(err) + ").");
  }
  const auto count = size / toBytes(dtype);
  const auto dev_ptr = reinterpret_cast<CUdeviceptr>(*data_ptr);
  switch (dtype)
  {
    case dType::Float: {
      auto cval = static_cast<float>(value);
      err = cuMemsetD32(dev_ptr, *(reinterpret_cast<uint32_t *>(&cval)), count);
      break;
    }
    case dType::Int64: {
      std::vector<int64_t> host_buffer(count, static_cast<int64_t>(value));
      writeMemory(device, data_ptr, size, host_buffer.data());
      break;
    }
    case dType::UInt64: {
      std::vector<uint64_t> host_buffer(count, static_cast<uint64_t>(value));
      writeMemory(device, data_ptr, size, host_buffer.data());
      break;
    }
    case dType::Int32: {
      auto cval = static_cast<int32_t>(value);
      err = cuMemsetD32(dev_ptr, *(reinterpret_cast<uint32_t *>(&cval)), count);
      break;
    }
    case dType::UInt32: {
      auto cval = static_cast<uint32_t>(value);
      err = cuMemsetD32(dev_ptr, *(reinterpret_cast<uint32_t *>(&cval)), count);
      break;
    }
    case dType::Int16: {
      auto cval = static_cast<int16_t>(value);
      err = cuMemsetD16(dev_ptr, *(reinterpret_cast<uint16_t *>(&cval)), count);
      break;
    }
    case dType::UInt16: {
      auto cval = static_cast<uint16_t>(value);
      err = cuMemsetD16(dev_ptr, *(reinterpret_cast<uint16_t *>(&cval)), count);
      break;
    }
    case dType::Int8: {
      auto cval = static_cast<int8_t>(value);
      err = cuMemsetD8(dev_ptr, *(reinterpret_cast<uint8_t *>(&cval)), count);
      break;
    }
    case dType::UInt8: {
      auto cval = static_cast<uint8_t>(value);
      err = cuMemsetD8(dev_ptr, *(reinterpret_cast<uint8_t *>(&cval)), count);
      break;
    }
    default:
      std::cerr << "Warning: Unsupported value size for cuda setMemory" << std::endl;
      break;
  }
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
                       const float &           value,
                       const dType &           dtype) const -> void
{
#if USE_CUDA
  auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
  auto err = cuCtxSetCurrent(cuda_device->getCUDAContext());
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error (cuda): Failed to get context from device (" + std::to_string(err) + ").");
  }

  switch (dtype)
  {
    case dType::Float: {
      using T = float;
      std::vector<T> host_buffer(width * height * depth, static_cast<T>(value));
      writeMemory(device, data_ptr, width, height, depth, toBytes(dtype), host_buffer.data());
      break;
    }
    case dType::Int32: {
      using T = int32_t;
      std::vector<T> host_buffer(width * height * depth, static_cast<T>(value));
      writeMemory(device, data_ptr, width, height, depth, toBytes(dtype), host_buffer.data());
      break;
    }
    case dType::UInt32: {
      using T = uint32_t;
      std::vector<T> host_buffer(width * height * depth, static_cast<T>(value));
      writeMemory(device, data_ptr, width, height, depth, toBytes(dtype), host_buffer.data());
      break;
    }
    case dType::Int16: {
      using T = int16_t;
      std::vector<T> host_buffer(width * height * depth, static_cast<T>(value));
      writeMemory(device, data_ptr, width, height, depth, toBytes(dtype), host_buffer.data());
      break;
    }
    case dType::UInt16: {
      using T = uint16_t;
      std::vector<T> host_buffer(width * height * depth, static_cast<T>(value));
      writeMemory(device, data_ptr, width, height, depth, toBytes(dtype), host_buffer.data());
      break;
    }
    case dType::Int8: {
      using T = int8_t;
      std::vector<T> host_buffer(width * height * depth, static_cast<T>(value));
      writeMemory(device, data_ptr, width, height, depth, toBytes(dtype), host_buffer.data());
      break;
    }
    case dType::UInt8: {
      using T = uint8_t;
      std::vector<T> host_buffer(width * height * depth, static_cast<T>(value));
      writeMemory(device, data_ptr, width, height, depth, toBytes(dtype), host_buffer.data());
      break;
    }
    default:
      std::cerr << "Warning: Unsupported value size for cuda setMemory" << std::endl;
      break;
  }


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
#if USE_CUDA
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
#if USE_CUDA
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
#if USE_CUDA
  auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
  auto err = cuCtxSetCurrent(cuda_device->getCUDAContext());
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to set CUDA device before memory allocation.");
  }

  nvrtcProgram prog;
  auto         res = nvrtcCreateProgram(&prog, kernel_source.c_str(), nullptr, 0, nullptr, nullptr);
  if (res != NVRTC_SUCCESS)
  {
    throw std::runtime_error("Error in creating program.");
  }
  res = nvrtcCompileProgram(prog, 0, nullptr);
  if (res != NVRTC_SUCCESS)
  {
    throw std::runtime_error("Error in Compiling program.");
  }
  size_t ptxSize;
  nvrtcGetPTXSize(prog, &ptxSize);
  std::vector<char> ptx(ptxSize);
  nvrtcGetPTX(prog, ptx.data());

  CUmodule cuModule;
  err = cuModuleLoadData(&cuModule, ptx.data());
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error in loading module.");
  }

  CUfunction cuFunction;
  err = cuModuleGetFunction(&cuFunction, cuModule, kernel_name.c_str());
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error in getting function.");
  }

  *(reinterpret_cast<CUfunction *>(kernel)) = cuFunction;
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
#if USE_CUDA
  // TODO
  auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
  auto err = cuCtxSetCurrent(cuda_device->getCUDAContext());
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to set CUDA device before memory allocation.");
  }

  CUfunction cuFunction;
  CUresult   error;
  try
  {
    buildKernel(device, kernel_source, kernel_name, &cuFunction);
  }
  catch (const std::exception & e)
  {
    throw std::runtime_error("Error: Failed to build kernel. \n\t > " + std::string(e.what()));
  }

  std::vector<void *> argsValues(args.size());
  argsValues = args;
  std::array<size_t, 3> block_size = toBlockDim(global_size);
  err = cuLaunchKernel(cuFunction,
                       global_size.data()[0],
                       global_size.data()[1],
                       global_size.data()[2],
                       block_size.data()[0],
                       block_size.data()[1],
                       block_size.data()[2],
                       0,
                       cuda_device->getCUDAStream(),
                       argsValues.data(),
                       nullptr);

  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error in launching kernel.");
  }
#else
  throw std::runtime_error("Error: CUDA backend is not enabled");
#endif
}

auto
CUDABackend::getPreamble() const -> std::string
{
  return kernel::preamble_cu;
}

auto
CUDABackend::toBlockDim(const std::array<size_t, 3> & global_size) const -> std::array<size_t, 3>
{
  // In general, we add the gridDim.x (gridDim.y & gridDim.z) to the problem size, subtract one and divide by the
  // gridDim.x (gridDim.y & gridDim.z). However, since we're taking the global_size, which represents the gridDim which,
  // in itself, is the shape of the array that represents the problem size, we get the following formulas:
  std::array<size_t, 3> block_size = { (global_size.data()[0] + global_size.data()[0] - 1) / global_size.data()[0],
                                       (global_size.data()[1] + global_size.data()[1] - 1) / global_size.data()[1],
                                       (global_size.data()[2] + global_size.data()[2] - 1) / global_size.data()[2] };

  // One can notice that the blockDim (block_size) will always be set to 1.

  return block_size;
}

} // namespace cle
