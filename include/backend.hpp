#ifndef __INCLUDE_BACKEND_HPP
#define __INCLUDE_BACKEND_HPP

#define CLE_CUDA 1
#define CLE_OPENCL 1

#if CLE_OPENCL
#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 300
#endif
#include <CL/opencl.hpp>
#endif

#if CLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif

#include "device.hpp"

namespace cle
{
    class Backend
    {
    public:
        enum class Type
        {
            CUDA = 1,
            OPENCL = 0
        };

        using DevicePtr = std::shared_ptr<cle::Device>;

        Backend() = default;
        virtual ~Backend() = default;

        [[nodiscard]] virtual auto getType() const -> Backend::Type = 0;
        [[nodiscard]] virtual auto getDevicesList(const std::string &type) const -> std::vector<std::string> = 0;
        [[nodiscard]] virtual auto getDevices(const std::string &type) const -> std::vector<DevicePtr> = 0;
        [[nodiscard]] virtual auto getDevice(const std::string &name, const std::string &type) const -> DevicePtr = 0;

        virtual auto allocateMemory(const DevicePtr &device, const size_t &size, void **data_ptr) const -> void = 0;
        virtual auto freeMemory(const DevicePtr &device, void **data_ptr) const -> void = 0;
        virtual auto writeMemory(const DevicePtr &device, void **data_ptr, const size_t &size, const void *host_ptr) const -> void = 0;
        virtual auto readMemory(const DevicePtr &device, const void **data_ptr, const size_t &size, void *host_ptr) const -> void = 0;

        friend auto operator<<(std::ostream &out, const Backend::Type &backend_type) -> std::ostream &
        {
            switch (backend_type)
            {
            case Backend::Type::CUDA:
                out << "CUDA";
                break;
            case Backend::Type::OPENCL:
                out << "OpenCL";
                break;
            }
            return out;
        }
    };

    class CUDABackend : public Backend
    {
    public:
        CUDABackend()
        {
#if !CLE_CUDA
            std::cerr << "Warning: Instanciating an CUDA Backend but CUDA is not enabled." << std::endl;
#endif
        }

        ~CUDABackend() override = default;

        [[nodiscard]] auto getDevices(const std::string &type) const -> std::vector<DevicePtr> override
        {
#if CLE_CUDA
            int deviceCount;
            auto error = cudaGetDeviceCount(&deviceCount);
            std::vector<std::shared_ptr<Device>> devices;
            for (int i = 0; i < deviceCount; i++)
            {
                devices.push_back(std::make_shared<CUDADevice>(i));
            }
            return devices;
#else
            throw std::runtime_error("CUDABackend::getDevices: CUDA is not enabled");
#endif
        }

        [[nodiscard]] auto getDevice(const std::string &name, const std::string &type) const -> DevicePtr override
        {
#if CLE_CUDA
            auto devices = getDevices(type);
            auto ite = std::find_if(devices.begin(), devices.end(),
                                    [&name](const DevicePtr &dev)
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
#else
            throw std::runtime_error("CUDABackend::getDevices: CUDA is not enabled");
#endif
        }

        [[nodiscard]] auto getDevicesList(const std::string &type) const -> std::vector<std::string> override
        {
#if CLE_CUDA
            auto devices = getDevices(type);
            std::vector<std::string> deviceList;
            for (int i = 0; i < devices.size(); i++)
            {
                deviceList.push_back(devices[i]->getName());
            }
            return deviceList;
#else
            throw std::runtime_error("CUDABackend::getDevices: CUDA is not enabled");
#endif
        }

        [[nodiscard]] auto getType() const -> Backend::Type override
        {
            return Backend::Type::CUDA;
        }

        auto allocateMemory(const DevicePtr &device, const size_t &size, void **data_ptr) const -> void override
        {
#if CLE_CUDA
            auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
            auto err = cudaSetDevice(cuda_device->getCUDADeviceIndex());
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to set CUDA device before memory allocation.");
            }
            err = cudaMalloc(data_ptr, size);
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to allocate CUDA memory.");
            }
#else
            throw std::runtime_error("CUDABackend::getDevices: CUDA is not enabled");
#endif
        }

        auto freeMemory(const DevicePtr &device, void **data_ptr) const -> void override
        {
#if CLE_CUDA
            auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
            auto err = cudaSetDevice(cuda_device->getCUDADeviceIndex());
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to set CUDA device before memory allocation.");
            }
            err = cudaFree(*data_ptr);
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to free CUDA memory.");
            }
#else
            throw std::runtime_error("CUDABackend::getDevices: CUDA is not enabled");
#endif
        }

        auto writeMemory(const DevicePtr &device, void **data_ptr, const size_t &size, const void *host_ptr) const -> void override
        {
#if CLE_CUDA
            auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
            auto err = cudaSetDevice(cuda_device->getCUDADeviceIndex());
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to set CUDA device before memory allocation.");
            }
            err = cudaMemcpy(*data_ptr, host_ptr, size, cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to write CUDA memory.");
            }
#else
            throw std::runtime_error("CUDABackend::getDevices: CUDA is not enabled");
#endif
        }

        auto readMemory(const DevicePtr &device, const void **data_ptr, const size_t &size, void *host_ptr) const -> void override
        {
#if CLE_CUDA
            auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
            auto err = cudaSetDevice(cuda_device->getCUDADeviceIndex());
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to set CUDA device before memory allocation.");
            }
            err = cudaMemcpy(host_ptr, *data_ptr, size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to read CUDA memory.");
            }
#else
            throw std::runtime_error("CUDABackend::getDevices: CUDA is not enabled");
#endif
        }
    };

    class OpenCLBackend : public Backend
    {
    public:
        OpenCLBackend()
        {
#if !CLE_OPENCL
            std::cerr << "Warning: Instanciating an OpenCL Backend but OpenCL is not enabled." << std::endl;
#endif
        }

        ~OpenCLBackend() override = default;

        [[nodiscard]] auto getDevices(const std::string &type) const -> std::vector<DevicePtr> override
        {
#if CLE_OPENCL
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);

            std::vector<DevicePtr> devices;
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
#else
            throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
        }

        [[nodiscard]] auto getDevice(const std::string &name, const std::string &type) const -> DevicePtr override
        {
#if CLE_OPENCL
            auto devices = getDevices(type);
            auto ite = std::find_if(devices.begin(), devices.end(),
                                    [&name](const DevicePtr &dev)
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
#else
            throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
        }

        [[nodiscard]] auto getDevicesList(const std::string &type) const -> std::vector<std::string> override
        {
#if CLE_OPENCL
            auto devices = getDevices(type);
            std::vector<std::string> deviceList;
            for (int i = 0; i < devices.size(); i++)
            {
                deviceList.push_back(devices[i]->getName());
            }
            return deviceList;
#else
            throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
        }

        [[nodiscard]] auto getType() const -> Backend::Type override
        {
            return Backend::Type::OPENCL;
        }

        auto allocateMemory(const DevicePtr &device, const size_t &size, void **data_ptr) const -> void override
        {
#if CLE_OPENCL
            cl_int err;
            auto opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);
            auto context = opencl_device->getCLContext();
            auto mem = clCreateBuffer(context.get(), CL_MEM_READ_WRITE, size, nullptr, &err);
            if (err != CL_SUCCESS)
            {
                throw std::runtime_error("Error: Failed to allocate OpenCL memory.");
            }
            *data_ptr = static_cast<void *>(new cl_mem(mem));
#else
            throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
        }

        auto freeMemory(const DevicePtr &device, void **data_ptr) const -> void override
        {
#if CLE_OPENCL
            auto *cl_mem_ptr = static_cast<cl_mem *>(*data_ptr);
            auto err = clReleaseMemObject(*cl_mem_ptr);
            if (err != CL_SUCCESS)
            {
                throw std::runtime_error("Error: Failed to free OpenCL memory.");
            }
#else
            throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
        }

        auto writeMemory(const DevicePtr &device, void **data_ptr, const size_t &size, const void *host_ptr) const -> void override
        {
#if CLE_OPENCL
            auto opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);
            auto queue = opencl_device->getCLCommandQueue();
            auto err = clEnqueueWriteBuffer(queue.get(), *static_cast<cl_mem *>(*data_ptr), CL_TRUE, 0, size, host_ptr, 0, nullptr, nullptr);
            if (err != CL_SUCCESS)
            {
                throw std::runtime_error("Error: Failed to write OpenCL memory.");
            }
#else
            throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
        }

        auto readMemory(const DevicePtr &device, const void **data_ptr, const size_t &size, void *host_ptr) const -> void override
        {
#if CLE_OPENCL
            auto opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);
            auto queue = opencl_device->getCLCommandQueue();
            auto err = clEnqueueReadBuffer(queue.get(), *static_cast<const cl_mem *>(*data_ptr), CL_TRUE, 0, size, host_ptr, 0, nullptr, nullptr);
            if (err != CL_SUCCESS)
            {
                throw std::runtime_error("Error: Failed to read OpenCL memory.");
            }
#else
            throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
        }
    };

    class BackendManager
    {
    public:
        static auto getInstance() -> BackendManager &
        {
            static BackendManager instance;
            return instance;
        }

        auto setBackend(bool useCUDA) -> void
        {
            if (useCUDA)
            {
                this->backend = std::make_unique<CUDABackend>();
            }
            else
            {
                this->backend = std::make_unique<OpenCLBackend>();
            }
        }

        [[nodiscard]] auto getBackend() const -> const Backend &
        {
            if (!this->backend)
            {
                throw std::runtime_error("Backend not selected.");
            }
            return *this->backend;
        }

        auto operator=(const BackendManager &) -> BackendManager & = delete;
        BackendManager(const BackendManager &) = delete;

    private:
        std::shared_ptr<Backend> backend;
        BackendManager() = default;
    };

} // namespace cle

#endif // __INCLUDE_BACKEND_HPP
