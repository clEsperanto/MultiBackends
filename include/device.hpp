#ifndef __INCLUDE_DEVICE_HPP
#define __INCLUDE_DEVICE_HPP

#include <iostream>
#include <map>
#include <sstream>

#if CLE_OPENCL
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#endif

#if CLE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif

namespace cle
{
    class Device
    {
    public:
        enum class Type
        {
            CUDA,
            OPENCL
        };

        Device() = default;
        virtual ~Device() = default;

        virtual auto initialize() -> void = 0;
        virtual auto finalize() -> void = 0;
        virtual auto finish() -> void = 0;

        [[nodiscard]] virtual auto isInitialized() const -> bool = 0;
        [[nodiscard]] virtual auto isAvailable() const -> bool = 0;
        [[nodiscard]] virtual auto getName() const -> std::string = 0;
        [[nodiscard]] virtual auto getInfo() const -> std::string = 0;
        [[nodiscard]] virtual auto getType() const -> Device::Type = 0;

        friend auto operator<<(std::ostream &out, const Device::Type &device_type) -> std::ostream &
        {
            switch (device_type)
            {
            case Device::Type::CUDA:
                out << "CUDA";
                break;
            case Device::Type::OPENCL:
                out << "OpenCL";
                break;
            }
            return out;
        }

        friend auto operator<<(std::ostream &out, const Device &device) -> std::ostream &
        {
            out << device.getName() << " (" << device.getType() << ")";
            return out;
        }
    };

#if CLE_OPENCL
    class OpenCLDevice : public Device
    {
    public:
        OpenCLDevice(const cl_platform_id &platform, const cl_device_id &device) : clDevice(device), clPlatform(platform) {}

        ~OpenCLDevice() override
        {
            if (isInitialized())
            {
                finalize();
            }
        }

        [[nodiscard]] auto getType() const -> Device::Type override
        {
            return Device::Type::OPENCL;
        }

        auto initialize() -> void override
        {
            if (isInitialized())
            {
                std::cerr << "OpenCL device already initialized" << std::endl;
                return;
            }
            cl_int err = CL_SUCCESS;
            clContext = clCreateContext(nullptr, 1, &clDevice, nullptr, nullptr, &err);
            if (err != CL_SUCCESS)
            {
                std::cerr << "Failed to create OpenCL context" << std::endl;
                return;
            }
            clCommandQueue = clCreateCommandQueueWithProperties(clContext, clDevice, nullptr, &err);
            if (err != CL_SUCCESS)
            {
                std::cerr << "Failed to create OpenCL command queue" << std::endl;
                return;
            }
            initialized = true;
        }

        auto finalize() -> void override
        {
            if (!isInitialized())
            {
                std::cerr << "OpenCL device not initialized" << std::endl;
                return;
            }
            this->finish();
            clReleaseContext(clContext);
            clReleaseCommandQueue(clCommandQueue);
            clReleaseDevice(clDevice);
            initialized = false;
        }

        auto finish() -> void override
        {
            if (!isInitialized())
            {
                std::cerr << "OpenCL device not initialized" << std::endl;
                return;
            }
            clFinish(clCommandQueue);
        }

        [[nodiscard]] auto isInitialized() const -> bool override
        {
            return initialized;
        }

        [[nodiscard]] auto isAvailable() const -> bool override
        {
            cl_bool available;
            clGetDeviceInfo(clDevice, CL_DEVICE_AVAILABLE, sizeof(cl_bool), &available, NULL);
            return static_cast<bool>(available);
        }

        [[nodiscard]] auto getCLPlatform() const -> const cl_platform_id &
        {
            return clPlatform;
        }

        [[nodiscard]] auto getCLDevice() const -> const cl_device_id &
        {
            return clDevice;
        }

        [[nodiscard]] auto getCLContext() const -> const cl_context &
        {
            return clContext;
        }

        [[nodiscard]] auto getCLCommandQueue() const -> const cl_command_queue &
        {
            return clCommandQueue;
        }

        [[nodiscard]] auto getName() const -> std::string override
        {
            char name[256];
            clGetDeviceInfo(clDevice, CL_DEVICE_NAME, sizeof(char) * 256, name, NULL);
            return std::string(name);
        }

        [[nodiscard]] auto getInfo() const -> std::string override
        {
            std::ostringstream result;
            char version[256];
            cl_device_type type;
            cl_uint compute_units;
            size_t global_mem_size;
            size_t max_mem_size;

            // Get device information
            const auto &name = getName();
            clGetDeviceInfo(clDevice, CL_DEVICE_VERSION, sizeof(char) * 256, &version, NULL);
            clGetDeviceInfo(clDevice, CL_DEVICE_TYPE, sizeof(cl_device_type), &type, NULL);
            clGetDeviceInfo(clDevice, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &compute_units, NULL);
            clGetDeviceInfo(clDevice, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(size_t), &global_mem_size, NULL);
            clGetDeviceInfo(clDevice, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_t), &max_mem_size, NULL);

            // Print device information to output string
            result << name << " (" << std::string(version) << ")\n";
            switch (type)
            {
            case CL_DEVICE_TYPE_CPU:
                result << "\tType: CPU\n";
                break;
            case CL_DEVICE_TYPE_GPU:
                result << "\tType: GPU\n";
                break;
            default:
                result << "\tType: Unknown\n";
                break;
            }
            result << "\tCompute Units: " << compute_units << '\n';
            result << "\tGlobal Memory Size: " << (global_mem_size / 1000000) << " MB\n";
            result << "\tMaximum Object Size: " << (max_mem_size / 1000000) << " MB\n";
            return result.str();
        }

        [[nodiscard]] auto getCache() -> std::map<std::string, cl_program> &
        {
            return this->cache;
        }

    private:
        cl_platform_id clPlatform;
        cl_device_id clDevice;
        cl_context clContext;
        cl_command_queue clCommandQueue;
        std::map<std::string, cl_program> cache;
        bool initialized = false;
    };
#endif // CLE_OPENCL

#if CLE_CUDA
    class CUDADevice : public Device
    {
    public:
        CUDADevice(int deviceIndex) : cudaDeviceIndex(deviceIndex) {}

        ~CUDADevice() override
        {
            if (isInitialized())
            {
                finalize();
            }
        }

        [[nodiscard]] auto getType() const -> Device::Type override
        {
            return Device::Type::CUDA;
        }

        auto initialize() -> void override
        {
            if (isInitialized())
            {
                std::cerr << "CUDA device already initialized" << std::endl;
                return;
            }
            auto err = cudaSetDevice(cudaDeviceIndex);
            if (err != cudaSuccess)
            {
                std::cerr << "Failed to set CUDA device" << std::endl;
                return;
            }
            err = cudaStreamCreate(&cudaStream);
            if (err != cudaSuccess)
            {
                std::cerr << "Failed to create CUDA stream" << std::endl;
                return;
            }
            initialized = true;
        }

        auto finalize() -> void override
        {
            if (!isInitialized())
            {
                std::cerr << "CUDA device not initialized" << std::endl;
                return;
            }
            cudaStreamSynchronize(cudaStream);
            cudaStreamDestroy(cudaStream);
            initialized = false;
        }

        auto finish() -> void override
        {
            if (!isInitialized())
            {
                std::cerr << "CUDA device not initialized" << std::endl;
                return;
            }

            cudaStreamSynchronize(cudaStream);
        }

        [[nodiscard]] auto isInitialized() const -> bool override
        {
            return initialized;
        }

        [[nodiscard]] auto isAvailable() const -> bool override
        {
            int ecc_enabled;
            cudaDeviceGetAttribute(&ecc_enabled, cudaDevAttrEccEnabled, cudaDeviceIndex);
            return static_cast<bool>(ecc_enabled);
        }

        [[nodiscard]] auto getCUDADeviceIndex() const -> int
        {
            return cudaDeviceIndex;
        }

        [[nodiscard]] auto getCUDAStream() const -> cudaStream_t
        {
            return cudaStream;
        }

        [[nodiscard]] auto getName() const -> std::string override
        {
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, cudaDeviceIndex);
            return prop.name;
        }

        [[nodiscard]] auto getInfo() const -> std::string override
        {
            std::ostringstream result;
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, cudaDeviceIndex);

            result << prop.name << " (" << prop.major << "." << prop.minor << ")\n";
            result << "\tType: " << (prop.integrated ? "Integrated" : "Discrete") << '\n';
            result << "\tCompute Units: " << prop.multiProcessorCount << '\n';
            result << "\tGlobal Memory Size: " << (prop.totalGlobalMem / 1000000) << " MB\n";
            // result << "\tMaximum Object Size: " << (prop.maxMemoryAllocationSize / 1000000) << " MB\n";

            return result.str();
        }

        [[nodiscard]] auto getCache() -> std::map<std::string, CUmodule> &
        {
            return this->cache;
        }

    private:
        int cudaDeviceIndex;
        cudaStream_t cudaStream;
        bool initialized = false;
        std::map<std::string, CUmodule> cache;
    };
#endif // CLE_CUDA

} // namespace cle

#endif // __INCLUDE_DEVICE_HPP
