#ifndef __INCLUDE_BACKEND_HPP
#define __INCLUDE_BACKEND_HPP

#include <optional>

#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 300
#endif
#include <CL/opencl.hpp>

#include <cuda_runtime.h>

#include "device.hpp"

namespace cle
{
    class Backend
    {
    public:
        enum class Type
        {
            CUDA,
            OPENCL
        };

        Backend() = default;
        virtual ~Backend() = default;

        virtual Backend::Type getType() const = 0;
        virtual std::vector<std::string> getDevicesList(const std::string &type = "all") const = 0;
        virtual std::vector<std::shared_ptr<Device>> getDevices(const std::string &type = "all") const = 0;
        virtual std::shared_ptr<Device> getDevice(const std::string &name, const std::string &type = "all") const = 0;

        virtual void allocateMemory(const std::shared_ptr<Device> &device, const size_t &size, void **data_ptr) const = 0;
        virtual void freeMemory(const std::shared_ptr<Device> &device, void **data_ptr) const = 0;
        virtual void writeMemory(const std::shared_ptr<Device> &device, void **data_ptr, const size_t &size, const void *host_ptr) const = 0;
        virtual void readMemory(const std::shared_ptr<Device> &device, const void **data_ptr, const size_t &size, void *host_ptr) const = 0;
    };

    class CUDABackend : public Backend
    {
    public:
        CUDABackend() = default;
        virtual ~CUDABackend() override = default;

        virtual std::vector<std::shared_ptr<Device>> getDevices(const std::string &type = "all") const
        {
            int deviceCount;
            cudaError_t error = cudaGetDeviceCount(&deviceCount);
            std::vector<std::shared_ptr<Device>> devices;
            for (int i = 0; i < deviceCount; i++)
            {
                devices.push_back(std::make_shared<CUDADevice>(i));
            }
            return devices;
        }

        virtual std::shared_ptr<Device> getDevice(const std::string &name, const std::string &type = "all") const
        {
            auto devices = getDevices(type);
            auto ite = std::find_if(devices.begin(), devices.end(),
                                    [&name](const std::shared_ptr<Device> &dev)
                                    {
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
        }

        virtual std::vector<std::string> getDevicesList(const std::string &type = "all") const
        {
            auto devices = getDevices(type);
            std::vector<std::string> deviceList;
            for (int i = 0; i < devices.size(); i++)
            {
                deviceList.push_back(devices[i]->getName());
            }
            return deviceList;
        }

        virtual Backend::Type getType() const
        {
            return Backend::Type::CUDA;
        }

        virtual void allocateMemory(const std::shared_ptr<Device> &device, const size_t &size, void **data_ptr) const
        {
            auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
            cudaError_t err = cudaSetDevice(cuda_device->getCUDADeviceIndex());
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to set CUDA device before memory allocation.");
            }
            err = cudaMalloc(data_ptr, size);
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to allocate CUDA memory.");
            }
        }

        virtual void freeMemory(const std::shared_ptr<Device> &device, void **data_ptr) const
        {
            auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
            cudaError_t err = cudaSetDevice(cuda_device->getCUDADeviceIndex());
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to set CUDA device before memory allocation.");
            }
            err = cudaFree(*data_ptr);
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to free CUDA memory.");
            }
        }

        virtual void writeMemory(const std::shared_ptr<Device> &device, void **data_ptr, const size_t &size, const void *host_ptr) const
        {
            auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
            cudaError_t err = cudaSetDevice(cuda_device->getCUDADeviceIndex());
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to set CUDA device before memory allocation.");
            }
            err = cudaMemcpy(*data_ptr, host_ptr, size, cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to write CUDA memory.");
            }
        }

        virtual void readMemory(const std::shared_ptr<Device> &device, const void **data_ptr, const size_t &size, void *host_ptr) const
        {
            auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
            cudaError_t err = cudaSetDevice(cuda_device->getCUDADeviceIndex());
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to set CUDA device before memory allocation.");
            }
            err = cudaMemcpy(host_ptr, *data_ptr, size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to read CUDA memory.");
            }
        }
    };

    class OpenCLBackend : public Backend
    {
    public:
        OpenCLBackend() = default;
        virtual ~OpenCLBackend() override = default;

        virtual std::vector<std::shared_ptr<Device>> getDevices(const std::string &type = "all") const
        {
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);

            std::vector<std::shared_ptr<Device>> devices;
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
                std::cerr << "\tdefault: fetching all devices." << std::endl;
            }
            for (int i = 0; i < platforms.size(); i++)
            {
                std::vector<cl::Device> clDevices;
                platforms[i].getDevices(deviceType, &clDevices);
                for (int j = 0; j < clDevices.size(); j++)
                {
                    devices.push_back(std::make_shared<OpenCLDevice>(clDevices[j]));
                }
            }
            return devices;
        }

        virtual std::shared_ptr<Device> getDevice(const std::string &name, const std::string &type = "all") const
        {
            auto devices = getDevices(type);
            auto ite = std::find_if(devices.begin(), devices.end(),
                                    [&name](const std::shared_ptr<Device> &dev)
                                    {
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
        }

        virtual std::vector<std::string> getDevicesList(const std::string &type = "all") const
        {
            auto devices = getDevices(type);
            std::vector<std::string> deviceList;
            for (int i = 0; i < devices.size(); i++)
            {
                deviceList.push_back(devices[i]->getName());
            }
            return deviceList;
        }

        virtual Backend::Type getType() const
        {
            return Backend::Type::OPENCL;
        }

        virtual void allocateMemory(const std::shared_ptr<Device> &device, const size_t &size, void **data_ptr) const
        {
            cl_int err;
            auto opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);
            cl::Context context = opencl_device->getCLContext();
            cl::Buffer buffer = cl::Buffer(context, CL_MEM_READ_WRITE, size, nullptr, &err);
            if (err != CL_SUCCESS)
            {
                throw std::runtime_error("Error: Failed to allocate OpenCL memory.");
            }
            *data_ptr = static_cast<void *>(new cl::Memory(buffer));
        }

        virtual void freeMemory(const std::shared_ptr<Device> &device, void **data_ptr) const
        {
            cl::Memory *cl_mem_ptr = static_cast<cl::Memory *>(*data_ptr);
            cl_int err = clReleaseMemObject(cl_mem_ptr->get());
            if (err != CL_SUCCESS)
            {
                throw std::runtime_error("Error: Failed to free OpenCL memory.");
            }
            delete cl_mem_ptr;
        }

        virtual void writeMemory(const std::shared_ptr<Device> &device, void **data_ptr, const size_t &size, const void *host_ptr) const
        {
            auto opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);
            cl::Context context = opencl_device->getCLContext();
            cl::CommandQueue queue = opencl_device->getCLCommandQueue();
            cl_int err = queue.enqueueWriteBuffer(*static_cast<cl::Buffer *>(*data_ptr), CL_TRUE, 0, size, host_ptr);
            if (err != CL_SUCCESS)
            {
                throw std::runtime_error("Error: Failed to write OpenCL memory.");
            }
        }

        virtual void readMemory(const std::shared_ptr<Device> &device, const void **data_ptr, const size_t &size, void *host_ptr) const
        {
            auto opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);
            cl::Context context = opencl_device->getCLContext();
            cl::CommandQueue queue = opencl_device->getCLCommandQueue();
            cl_int err = queue.enqueueReadBuffer(*static_cast<const cl::Buffer *>(*data_ptr), CL_TRUE, 0, size, host_ptr);
            if (err != CL_SUCCESS)
            {
                throw std::runtime_error("Error: Failed to read OpenCL memory.");
            }
        }
    };

    class BackendManager
    {
    public:
        static BackendManager &getInstance()
        {
            static BackendManager instance;
            return instance;
        }

        void selectBackend(bool useCUDA)
        {
            if (useCUDA)
            {
                backend = std::make_unique<CUDABackend>();
            }
            else
            {
                backend = std::make_unique<OpenCLBackend>();
            }
        }

        Backend &getBackend() const
        {
            if (!backend)
            {
                throw std::runtime_error("Backend not selected.");
            }
            return *backend;
        }

        BackendManager &operator=(const BackendManager &) = delete;
        BackendManager(const BackendManager &) = delete;

    private:
        std::shared_ptr<Backend> backend;
        BackendManager() {}
    };

} // namespace cle

#endif // __INCLUDE_BACKEND_HPP
