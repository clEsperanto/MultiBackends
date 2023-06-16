#include "device.hpp"

namespace cle
{

#if CLE_CUDA

CUDADevice::CUDADevice(int deviceIndex)
  : cudaDeviceIndex(deviceIndex)
{
  initialize();
}

CUDADevice::~CUDADevice()
{
  if (isInitialized())
  {
    finalize();
  }
}

auto
CUDADevice::getType() const -> Device::Type
{
  return Device::Type::CUDA;
}

auto
CUDADevice::initialize() -> void
{
  if (isInitialized())
  {
    std::cerr << "CUDA device already initialized" << std::endl;
    return;
  }
  auto err = cuDeviceGet(&cudaDevice, 0);
  if (err != CUDA_SUCCESS)
  {
    std::cerr << "Failed to get CUDA device" << std::endl;
    return;
  }
  err = cuCtxCreate(&cudaContext, 0, cudaDevice);
  if (err != CUDA_SUCCESS)
  {
    std::cerr << "Failed to create CUDA context" << std::endl;
    return;
  }
  err = cuStreamCreate(&cudaStream, CU_STREAM_DEFAULT);
  if (err != CUDA_SUCCESS)
  {
    std::cerr << "Failed to create CUDA stream" << std::endl;
    return;
  }
  initialized = true;
}

auto
CUDADevice::finalize() -> void
{
  if (!isInitialized())
  {
    std::cerr << "CUDA device not initialized" << std::endl;
    return;
  }
  finish();
  cuStreamDestroy(cudaStream);
  cuCtxDestroy(cudaContext);
  cuCtxSetCurrent(nullptr);
  initialized = false;
}

auto
CUDADevice::finish() -> void
{
  if (!isInitialized())
  {
    std::cerr << "CUDA device not initialized" << std::endl;
    return;
  }
  cuStreamSynchronize(cudaStream);
}

auto
CUDADevice::isInitialized() const -> bool
{
  return initialized;
}

auto
CUDADevice::getCUDADeviceIndex() const -> int
{
  return cudaDeviceIndex;
}

auto
CUDADevice::getCUDADevice() const -> const CUdevice &
{
  return cudaDevice;
}

auto
CUDADevice::getCUDAContext() const -> const CUcontext &
{
  return cudaContext;
}

auto
CUDADevice::getCUDAStream() const -> const CUstream &
{
  return cudaStream;
}

auto
CUDADevice::getName() const -> std::string
{
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, cudaDeviceIndex);
  return prop.name;
}

auto
CUDADevice::getInfo() const -> std::string
{
  std::ostringstream result;
  cudaDeviceProp     prop{};
  cudaGetDeviceProperties(&prop, cudaDeviceIndex);

  result << static_cast<char *>(prop.name) << " (" << prop.major << "." << prop.minor << ")\n";
  result << "\tType: " << (prop.integrated != 0 ? "Integrated" : "Discrete") << '\n';
  result << "\tCompute Units: " << prop.multiProcessorCount << '\n';
  result << "\tGlobal Memory Size: " << (prop.totalGlobalMem / 1000000) << " MB\n";
  // result << "\tMaximum Object Size: " << (prop.maxMemoryAllocationSize / 1000000) << " MB\n";

  return result.str();
}

auto
CUDADevice::getCache() -> std::map<std::string, CUmodule> &
{
  return this->cache;
}

#endif // CLE_CUDA

} // namespace cle
