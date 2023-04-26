#ifndef __INCLUDE_DEVICE_HPP
#define __INCLUDE_DEVICE_HPP

#include <iostream>
#include <sstream>

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
        OpenCLDevice(const cl::Device &device) : clDevice(device) {}

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
            clContext = cl::Context({clDevice}, NULL, NULL, NULL, &err);
            if (err != CL_SUCCESS)
            {
                std::cerr << "Failed to create OpenCL context" << std::endl;
                return;
            }
            clCommandQueue = cl::CommandQueue(clContext, clDevice, 0, &err);
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
            clCommandQueue.finish();
            initialized = false;
        }

        auto finish() -> void override
        {
            if (!isInitialized())
            {
                std::cerr << "OpenCL device not initialized" << std::endl;
                return;
            }
            clCommandQueue.finish();
        }

        [[nodiscard]] auto isInitialized() const -> bool override
        {
            return initialized;
        }

        [[nodiscard]] auto getCLPlatform() const -> const cl::Platform &
        {
            return clPlatform;
        }

        [[nodiscard]] auto getCLDevice() const -> const cl::Device &
        {
            return clDevice;
        }

        [[nodiscard]] auto getCLContext() const -> const cl::Context &
        {
            return clContext;
        }

        [[nodiscard]] auto getCLCommandQueue() const -> const cl::CommandQueue &
        {
            return clCommandQueue;
        }

        [[nodiscard]] auto getName() const -> std::string override
        {
            return clDevice.getInfo<CL_DEVICE_NAME>();
        }

        [[nodiscard]] auto getInfo() const -> std::string override
        {
            std::ostringstream result;
            std::string version;
            cl_device_type type;
            cl_uint compute_units;
            size_t global_mem_size;
            size_t max_mem_size;

            // Get device information
            const auto &name = getName();
            clDevice.getInfo(CL_DEVICE_VERSION, &version);
            clDevice.getInfo(CL_DEVICE_TYPE, &type);
            clDevice.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &compute_units);
            clDevice.getInfo(CL_DEVICE_GLOBAL_MEM_SIZE, &global_mem_size);
            clDevice.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &max_mem_size);

            // Print device information to output string
            result << name << " (" << version << ")\n";
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

    private:
        cl::Platform clPlatform;
        cl::Device clDevice;
        cl::Context clContext;
        cl::CommandQueue clCommandQueue;
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

    private:
        int cudaDeviceIndex;
        cudaStream_t cudaStream;
        bool initialized = false;
    };
#endif // CLE_CUDA

} // namespace cle

#endif // __INCLUDE_DEVICE_HPP
