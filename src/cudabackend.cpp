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
    throw std::runtime_error("Error: Failed to get CUDA device count.");
  }
  std::vector<Device::Pointer> devices;
  for (int i = 0; i < deviceCount; i++)
  {
    devices.push_back(std::make_shared<CUDADevice>(i));
  }
  return devices;
#else
  throw std::runtime_error("Error: CUDA backend is not enabled");
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
    return std::move(devices.back());
  }
  return nullptr;
#else
  throw std::runtime_error("Error: CUDA backend is not enabled");
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
  throw std::runtime_error("Error: CUDA backend is not enabled");
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
    throw std::runtime_error("Error: Failed to get CUDA context before memory allocation.");
  }
  CUdeviceptr mem;
  err = cuMemAlloc(&mem, size);
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to allocate CUDA memory.");
  }
  *data_ptr = reinterpret_cast<void *>(mem);
#else
  throw std::runtime_error("Error: CUDA backend is not enabled");
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
  // TODO
#else
  throw std::runtime_error("Error: CUDA backend is not enabled");
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
    throw std::runtime_error("Error: Failed to set CUDA device before memory allocation.");
  }
  err = cuMemFree(reinterpret_cast<CUdeviceptr>(*data_ptr));
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to free CUDA memory.");
  }
#else
  throw std::runtime_error("Error: CUDA backend is not enabled");
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
    throw std::runtime_error("Error: Failed to set CUDA device before memory allocation.");
  }
  err = cuMemcpyHtoD(reinterpret_cast<CUdeviceptr>(*data_ptr), host_ptr, size);
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to write CUDA memory.");
  }
#else
  throw std::runtime_error("Error: CUDA backend is not enabled");
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
  // TODO
#else
  throw std::runtime_error("Error: CUDA backend is not enabled");
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
    throw std::runtime_error("Error: Failed to set CUDA device before memory allocation.");
  }
  err = cuMemcpyDtoH(host_ptr, reinterpret_cast<CUdeviceptr>(*data_ptr), size);
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to read CUDA memory.");
  }
#else
  throw std::runtime_error("Error: CUDA backend is not enabled");
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
  // TODO
#else
  throw std::runtime_error("Error: CUDA backend is not enabled");
#endif
}

auto
CUDABackend::copyMemory(const Device::Pointer & device,
                        const void **           src_data_ptr,
                        const size_t &          size,
                        void **                 dst_data_ptr) const -> void
{
#if CLE_CUDA
  auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
  auto err = cuCtxSetCurrent(cuda_device->getCUDAContext());
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to set CUDA device before memory allocation.");
  }
  err = cuMemcpyDtoD(reinterpret_cast<CUdeviceptr>(*dst_data_ptr), reinterpret_cast<CUdeviceptr>(*src_data_ptr), size);
  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to write CUDA memory " + std::to_string(err));
  }
#else
  throw std::runtime_error("Error: CUDA backend is not enabled");
#endif
}

auto
CUDABackend::copyMemory(const Device::Pointer & device,
                        const void **           src_data_ptr,
                        const size_t &          width,
                        const size_t &          height,
                        const size_t &          depth,
                        const size_t &          bytes,
                        void **                 dst_data_ptr) const -> void
{
#if CLE_CUDA
  // TODO
#else
  throw std::runtime_error("Error: CUDA backend is not enabled");
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
    throw std::runtime_error("Error: Failed to set CUDA device before memory allocation.");
  }


  // TODO

  if (err != CUDA_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to set CUDA memory.");
  }
#else
  throw std::runtime_error("Error: CUDA backend is not enabled");
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
  // TODO
#else
  throw std::runtime_error("Error: CUDA backend is not enabled");
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
  throw std::runtime_error("Error: CUDA backend is not enabled");
#endif
}

auto
CUDABackend::saveProgramToCache(const Device::Pointer & device, const std::string & hash, void * program) const -> void
{
#if CLE_CUDA
  auto cuda_device = std::dynamic_pointer_cast<CUDADevice>(device);
  cuda_device->getCache().emplace_hint(cuda_device->getCache().end(), hash, reinterpret_cast<CUmodule>(program));
#else
  throw std::runtime_error("Error: CUDA backend is not enabled");
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
  throw std::runtime_error("Error: CUDA backend is not enabled");
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
  return kernel::preamble_cu; // @StRigaud TODO: add cuda preamble from header file
}

} // namespace cle
