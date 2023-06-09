#include "backend.hpp"
#include "cle_preamble_cl.h"

#include <array>

namespace cle
{

auto
OpenCLBackend::getDevices(const std::string & type) const -> std::vector<DevicePtr>
{
#if CLE_OPENCL

  std::vector<DevicePtr> devices; // set device type

  cl_uint platformCount = 0;
  clGetPlatformIDs(0, nullptr, &platformCount); // get number of platforms
  if (platformCount == 0)
  {
    throw std::runtime_error("Error: Failed to find OpenCL platform.");
  }

  std::vector<cl_platform_id> platformIds(platformCount);
  clGetPlatformIDs(platformCount, platformIds.data(), nullptr); // get platform ids

  cl_device_type deviceType;
  if (type == "cpu")
  {
    deviceType = CL_DEVICE_TYPE_CPU;
  }
  else if (type == "gpu")
  {
    deviceType = CL_DEVICE_TYPE_GPU;
  }
  else if (type == "all")
  {
    deviceType = CL_DEVICE_TYPE_ALL;
  }
  else
  {
    std::cerr << "Warning: Unknown device type '" << type << "' provided." << std::endl;
    std::cerr << "\tdefault: fetching 'all' devices type." << std::endl;
    deviceType = CL_DEVICE_TYPE_ALL;
  }

  for (auto && platform_id : platformIds) // for each platform
  {
    cl_uint deviceCount = 0;
    clGetDeviceIDs(platform_id, deviceType, 0, nullptr, &deviceCount); // get number of devices

    if (deviceCount == 0)
    {
      continue;
    }

    std::vector<cl_device_id> deviceIds(deviceCount);
    clGetDeviceIDs(platform_id, deviceType, deviceCount, deviceIds.data(), nullptr); // get device ids

    for (auto && device_id : deviceIds)                                              // for each device
    {
      devices.emplace_back(std::make_shared<OpenCLDevice>(platform_id, device_id));
    }
  }

  if (devices.empty())
  {
    throw std::runtime_error("Error: Failed to find OpenCL device.");
  }

  return devices;
#else
  throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
}

auto
OpenCLBackend::getDevice(const std::string & name, const std::string & type) const -> DevicePtr
{
#if CLE_OPENCL
  auto devices = getDevices(type);
  auto ite = std::find_if(devices.begin(), devices.end(), [&name](const DevicePtr & dev) {
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
  throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
}

auto
OpenCLBackend::getDevicesList(const std::string & type) const -> std::vector<std::string>
{
#if CLE_OPENCL
  auto                     devices = getDevices(type);
  std::vector<std::string> deviceList;
  for (auto && device : devices)
  {
    deviceList.emplace_back(device->getName());
  }
  return deviceList;
#else
  throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
}

auto
OpenCLBackend::getType() const -> Backend::Type
{
  return Backend::Type::OPENCL;
}

auto
OpenCLBackend::allocateMemory(const DevicePtr & device, const size_t & size, void ** data_ptr) const -> void
{
#if CLE_OPENCL
  cl_int err;
  auto   opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);
  auto   mem = clCreateBuffer(opencl_device->getCLContext(), CL_MEM_READ_WRITE, size, nullptr, &err);
  if (err != CL_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to allocate OpenCL memory.");
  }
  *data_ptr = static_cast<void *>(new cl_mem(mem));
#else
  throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
}

auto
OpenCLBackend::allocateMemory(const DevicePtr & device,
                              const size_t &    width,
                              const size_t &    height,
                              const size_t &    depth,
                              const dType &     dtype,
                              void **           data_ptr) const -> void
{
#if CLE_OPENCL
  allocateMemory(device, width * height * depth * toBytes(dtype), data_ptr);

  // auto opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);

  // cl_image_format image_format = { 0 };
  // image_format.image_channel_order = CL_INTENSITY;
  // cl_image_desc image_desc = { 0 };
  // image_desc.image_width = width;
  // image_desc.image_height = height;
  // image_desc.image_depth = depth;
  // image_desc.image_row_pitch = 0;
  // image_desc.image_slice_pitch = 0;
  // image_desc.num_mip_levels = 0;
  // image_desc.num_samples = 0;
  // image_desc.buffer = nullptr;
  // switch (dtype)
  // {
  //   case dType::Float:
  //     image_format.image_channel_data_type = CL_FLOAT;
  //     break;
  //   case dType::Int32:
  //     image_format.image_channel_data_type = CL_SIGNED_INT32;
  //     break;
  //   case dType::UInt32:
  //     image_format.image_channel_data_type = CL_UNSIGNED_INT32;
  //     break;
  //   case dType::Int8:
  //     image_format.image_channel_data_type = CL_SIGNED_INT8;
  //     break;
  //   case dType::UInt8:
  //     image_format.image_channel_data_type = CL_UNSIGNED_INT8;
  //     break;
  //   case dType::Int16:
  //     image_format.image_channel_data_type = CL_SIGNED_INT16;
  //     break;
  //   case dType::UInt16:
  //     image_format.image_channel_data_type = CL_UNSIGNED_INT16;
  //     break;
  //   default:
  //     image_format.image_channel_data_type = CL_FLOAT;
  //     std::cerr << "WARNING: Unsupported data type for Image. Default type float is used." << std::endl;
  //     break;
  // }
  // if (depth > 1)
  // {
  //   image_desc.image_type = CL_MEM_OBJECT_IMAGE3D;
  // }
  // else if (height > 1)
  // {
  //   image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
  // }
  // else
  // {
  //   image_desc.image_type = CL_MEM_OBJECT_IMAGE1D;
  // }
  // cl_int err;
  // auto   image =
  //   clCreateImage(opencl_device->getCLContext(), CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
  // if (err != CL_SUCCESS)
  // {
  //   throw std::runtime_error("Error: Failed to allocate OpenCL memory.");
  // }
  // *data_ptr = static_cast<void *>(new cl_mem(image));
#else
  throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
}

auto
OpenCLBackend::freeMemory(const DevicePtr & device, const mType & mtype, void ** data_ptr) const -> void
{
#if CLE_OPENCL
  auto * cl_mem_ptr = static_cast<cl_mem *>(*data_ptr);
  auto   err = clReleaseMemObject(*cl_mem_ptr);
  if (err != CL_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to free OpenCL memory.");
  }
#else
  throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
}

auto
OpenCLBackend::writeMemory(const DevicePtr & device, void ** data_ptr, const size_t & size, const void * host_ptr) const
  -> void
{
#if CLE_OPENCL
  auto opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);
  auto err = clEnqueueWriteBuffer(opencl_device->getCLCommandQueue(),
                                  *static_cast<cl_mem *>(*data_ptr),
                                  CL_TRUE,
                                  0,
                                  size,
                                  host_ptr,
                                  0,
                                  nullptr,
                                  nullptr);
  if (err != CL_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to write OpenCL memory.");
  }
#else
  throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
}

auto
OpenCLBackend::writeMemory(const DevicePtr & device,
                           void **           data_ptr,
                           const size_t &    width,
                           const size_t &    height,
                           const size_t &    depth,
                           const size_t &    bytes,
                           const void *      host_ptr) const -> void
{
#if CLE_OPENCL
  auto                        opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);
  const std::array<size_t, 3> origin = { 0, 0, 0 };
  const std::array<size_t, 3> region = { width * bytes, height, depth };
  auto                        err = clEnqueueWriteBufferRect(opencl_device->getCLCommandQueue(),
                                      *static_cast<cl_mem *>(*data_ptr),
                                      CL_TRUE,
                                      origin.data(),
                                      origin.data(),
                                      region.data(),
                                      0,
                                      0,
                                      0,
                                      0,
                                      host_ptr,
                                      0,
                                      nullptr,
                                      nullptr);
  if (err != CL_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to write OpenCL memory.");
  }
#else
  throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
}

auto
OpenCLBackend::readMemory(const DevicePtr & device, const void ** data_ptr, const size_t & size, void * host_ptr) const
  -> void
{
#if CLE_OPENCL
  auto opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);
  auto err = clEnqueueReadBuffer(opencl_device->getCLCommandQueue(),
                                 *static_cast<const cl_mem *>(*data_ptr),
                                 CL_TRUE,
                                 0,
                                 size,
                                 host_ptr,
                                 0,
                                 nullptr,
                                 nullptr);
  if (err != CL_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to read OpenCL memory.");
  }
#else
  throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
}

auto
OpenCLBackend::readMemory(const DevicePtr & device,
                          const void **     data_ptr,
                          const size_t &    width,
                          const size_t &    height,
                          const size_t &    depth,
                          const size_t &    bytes,
                          void *            host_ptr) const -> void
{
#if CLE_OPENCL
  auto                        opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);
  const std::array<size_t, 3> origin = { 0, 0, 0 };
  const std::array<size_t, 3> region = { width * bytes, height, depth };
  auto                        err = clEnqueueReadBufferRect(opencl_device->getCLCommandQueue(),
                                     *static_cast<const cl_mem *>(*data_ptr),
                                     CL_TRUE,
                                     origin.data(),
                                     origin.data(),
                                     region.data(),
                                     0,
                                     0,
                                     0,
                                     0,
                                     host_ptr,
                                     0,
                                     nullptr,
                                     nullptr);
  if (err != CL_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to read OpenCL memory.");
  }
#else
  throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
}

auto
OpenCLBackend::copyMemory(const DevicePtr & device,
                          const void **     src_data_ptr,
                          const size_t &    size,
                          void **           dst_data_ptr) const -> void
{
#if CLE_OPENCL
  auto opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);
  auto err = clEnqueueCopyBuffer(opencl_device->getCLCommandQueue(),
                                 *static_cast<const cl_mem *>(*src_data_ptr),
                                 *static_cast<cl_mem *>(*dst_data_ptr),
                                 0,
                                 0,
                                 size,
                                 0,
                                 nullptr,
                                 nullptr);
  if (err != CL_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to write OpenCL memory.");
  }
#else
  throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
}

auto
OpenCLBackend::copyMemory(const DevicePtr & device,
                          const void **     src_data_ptr,
                          const size_t &    width,
                          const size_t &    height,
                          const size_t &    depth,
                          const size_t &    bytes,
                          void **           dst_data_ptr) const -> void
{
#if CLE_OPENCL
  auto                        opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);
  const std::array<size_t, 3> origin = { 0, 0, 0 };
  const std::array<size_t, 3> region = { width * bytes, height, depth };
  auto                        err = clEnqueueCopyBufferRect(opencl_device->getCLCommandQueue(),
                                     *static_cast<const cl_mem *>(*src_data_ptr),
                                     *static_cast<cl_mem *>(*dst_data_ptr),
                                     origin.data(),
                                     origin.data(),
                                     region.data(),
                                     0,
                                     0,
                                     0,
                                     0,
                                     0,
                                     nullptr,
                                     nullptr);
  if (err != CL_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to write OpenCL memory.");
  }
#else
  throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
}

auto
OpenCLBackend::setMemory(const DevicePtr & device,
                         void **           data_ptr,
                         const size_t &    size,
                         const void *      value,
                         const size_t &    value_size) const -> void
{
#if CLE_OPENCL
  auto opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);
  auto err = clEnqueueFillBuffer(opencl_device->getCLCommandQueue(),
                                 *static_cast<cl_mem *>(*data_ptr),
                                 value,
                                 value_size,
                                 0,
                                 size,
                                 0,
                                 nullptr,
                                 nullptr);
  if (err != CL_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to set OpenCL memory with error code " + std::to_string(err) + ".");
  }
#else
  throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
}

auto
OpenCLBackend::setMemory(const DevicePtr & device,
                         void **           data_ptr,
                         const size_t &    width,
                         const size_t &    height,
                         const size_t &    depth,
                         const size_t &    bytes,
                         const void *      value) const -> void
{
#if CLE_OPENCL
  auto opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);
  auto err = clEnqueueFillBuffer(opencl_device->getCLCommandQueue(),
                                 *static_cast<cl_mem *>(*data_ptr),
                                 value,
                                 bytes,
                                 0,
                                 width * height * depth * bytes,
                                 0,
                                 nullptr,
                                 nullptr);
  if (err != CL_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to set OpenCL memory.");
  }
#else
  throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
}

auto
OpenCLBackend::loadProgramFromCache(const DevicePtr & device, const std::string & hash, void * program) const -> void
{
#if CLE_OPENCL
  auto opencl_device = std::dynamic_pointer_cast<OpenCLDevice>(device);

  cl_program prog = nullptr;
  auto       ite = opencl_device->getCache().find(hash);
  if (ite != opencl_device->getCache().end())
  {
    prog = ite->second;
  }
  *static_cast<cl_program *>(program) = prog;
#else
  throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
}

auto
OpenCLBackend::saveProgramToCache(const DevicePtr & device, const std::string & hash, void * program) const -> void
{

#if CLE_OPENCL
  auto opencl_device = std::dynamic_pointer_cast<OpenCLDevice>(device);
  opencl_device->getCache().emplace_hint(opencl_device->getCache().end(), hash, *static_cast<cl_program *>(program));
#else
  throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
}

auto
OpenCLBackend::buildKernel(const DevicePtr &   device,
                           const std::string & kernel_source,
                           const std::string & kernel_name,
                           cl_kernel &         kernel) const -> void
{
#if CLE_OPENCL
  std::cout << "\tbackend buildKernel START" << std::endl;
  std::cout << "\t\tbuilding kernel : " << kernel_name << std::endl;
  cl_int     err;
  auto       opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);
  cl_program prog = nullptr;
  // // std::string hash = std::to_string(std::hash<std::string>{}(kernel_source));
  // // loadProgramFromCache(device, hash, prog);
  // // if (prog == nullptr)
  // // {
  const char * source = kernel_source.c_str();
  prog = clCreateProgramWithSource(opencl_device->getCLContext(), 1, &source, nullptr, &err);
  if (err != CL_SUCCESS)
  {
    size_t                 len;
    std::array<char, 2048> buffer;
    clGetProgramBuildInfo(prog, opencl_device->getCLDevice(), CL_PROGRAM_BUILD_LOG, buffer.size(), buffer.data(), &len);
    std::cerr << buffer.data() << std::endl;
    throw std::runtime_error("Error: Failed to build OpenCL program.");
  }
  else
  {
    std::cout << "\t\tbuildKernel: clCreateProgramWithSource success" << std::endl;
  }
  cl_int buildStatus = clBuildProgram(prog, 0, nullptr, nullptr, nullptr, nullptr);
  if (buildStatus != CL_SUCCESS)
  {
    // Handle build error, e.g., retrieve build log using clGetProgramBuildInfo
    size_t logSize;
    clGetProgramBuildInfo(prog, opencl_device->getCLDevice(), CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
    char * buildLog = new char[logSize];
    clGetProgramBuildInfo(prog, opencl_device->getCLDevice(), CL_PROGRAM_BUILD_LOG, logSize, buildLog, nullptr);
    // Process and display the build log as needed
    delete[] buildLog;
  }
  else
  {
    std::cout << "\t\tbuildKernel: clBuildProgram success" << std::endl;
  }
  // // saveProgramToCache(device, hash, prog);
  // // }
  kernel = clCreateKernel(prog, kernel_name.c_str(), &err);
  if (err != CL_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to create OpenCL kernel (" + std::to_string(err) + ").)");
  }
  std::cout << "\tbackend buildKernel END" << std::endl;
#else
  throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
}

auto
OpenCLBackend::executeKernel(const DevicePtr &             device,
                             const std::string &           kernel_source,
                             const std::string &           kernel_name,
                             const std::array<size_t, 3> & global_size,
                             const std::vector<void *> &   args,
                             const std::vector<size_t> &   sizes) const -> void
{
  std::cout << "backend executeKernel START" << std::endl;
  std::cout << "\texecuting : " << kernel_name << std::endl;

  auto opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);

  // build kernel from source
  cl_kernel ocl_kernel;
  try
  {
    buildKernel(device, kernel_source, kernel_name, ocl_kernel);
  }
  catch (const std::exception & e)
  {
    throw std::runtime_error("Error while building kernel : " + std::string(e.what()));
  }

  // set kernel arguments
  for (size_t i = 0; i < args.size(); i++)
  {
    auto err = clSetKernelArg(ocl_kernel, i, sizes[i], args[i]);
    if (err != CL_SUCCESS)
    {
      throw std::runtime_error("Error: Failed to set OpenCL kernel arguments  (" + std::to_string(err) + ").)");
    }
  }

  // execute kernel
  auto err = clEnqueueNDRangeKernel(
    opencl_device->getCLCommandQueue(), ocl_kernel, 3, nullptr, global_size.data(), nullptr, 0, nullptr, nullptr);
  if (err != CL_SUCCESS)
  {
    throw std::runtime_error("Error: Failed to execute OpenCL kernel(" + std::to_string(err) + ").)");
  }
  std::cout << "backend executeKernel END" << std::endl;
}

auto
OpenCLBackend::getPreamble() const -> std::string
{
  return kernel::preamble_cl; // @StRigaud TODO: add OpenCL preamble from header file
}

} // namespace cle