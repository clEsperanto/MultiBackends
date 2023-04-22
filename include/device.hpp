#ifndef __INCLUDE_DEVICE_HPP
#define __INCLUDE_DEVICE_HPP

#include <iostream>

#include <CL/opencl.hpp>
#include <cuda_runtime.h>

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

        virtual void initialize() = 0;
        virtual void finalize() = 0;
        virtual void finish() = 0;

        virtual bool isInitialized() const = 0;
        virtual std::string getName() const = 0;
        virtual std::string getInfo() const = 0;
        virtual DeviceType getType() const = 0;
    };

    class OpenCLDevice : public Device
    {
    public:
        OpenCLDevice(const cl::Platform &platform,
                     const cl::Device &device) : clPlatform(platform),
                                                 clDevice(device) {}

        ~OpenCLDevice() override
        {
            if (isInitialized())
            {
                finalize();
            }
        }

        DeviceType getType() const override
        {
            return Device::Type::OPENCL;
        }

        void initialize() override
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
                std::cerr << "Failed to create OpenCL context: "
                          << cl::getOpenCLErrorCodeStr(err) << std::endl;
                return;
            }

            clCommandQueue = cl::CommandQueue(clContext, clDevice, 0, &err);
            if (err != CL_SUCCESS)
            {
                std::cerr << "Failed to create OpenCL command queue: "
                          << cl::getOpenCLErrorCodeStr(err) << std::endl;
                return;
            }

            initialized = true;
        }

        void finalize() override
        {
            if (!isInitialized())
            {
                std::cerr << "OpenCL device not initialized" << std::endl;
                return;
            }

            clCommandQueue.finish();
            initialized = false;
        }

        void finish() override
        {
            if (!isInitialized())
            {
                std::cerr << "OpenCL device not initialized" << std::endl;
                return;
            }

            clCommandQueue.finish();
        }

        bool isInitialized() const override
        {
            return initialized;
        }

        cl::Platform &getCLPlatform()
        {
            return clPlatform;
        }

        cl::Device &getCLDevice()
        {
            return clDevice;
        }

        cl::Context &getCLContext()
        {
            return clContext;
        }

        cl::CommandQueue &getCLCommandQueue()
        {
            return clCommandQueue;
        }

        std::string getName() const override
        {
            return clDevice.getInfo<CL_DEVICE_NAME>();
        }


        std::string getInfo() const override
        {
            std::ostringstream result;
            std::string        version;
            cl_device_type     type;
            cl_uint            compute_units;
            size_t             global_mem_size;
            size_t             max_mem_size;

            // Get device information
            const auto & name = getName(device_pointer);
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

        DeviceType getType() const override
        {
            return Device::Type::CUDA;
        }

        void initialize() override
        {
            if (isInitialized())
            {
                std::cerr << "CUDA device already initialized" << std::endl;
                return;
            }

            cudaError_t err = cudaSetDevice(cudaDeviceIndex);
            if (err != cudaSuccess)
            {
                std::cerr << "Failed to set CUDA device: "
                          << cudaGetErrorString(err) << std::endl;
                return;
            }

            err = cudaStreamCreate(&cudaStream);
            if (err != cudaSuccess)
            {
                std::cerr << "Failed to create CUDA stream: "
                          << cudaGetErrorString(err) << std::endl;
                return;
            }

            initialized = true;
        }

        void finalize() override
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

        void finish() override
        {
            if (!isInitialized())
            {
                std::cerr << "CUDA device not initialized" << std::endl;
                return;
            }

            cudaStreamSynchronize(cudaStream);
        }

        bool isInitialized() const override
        {
            return initialized;
        }

        int getCUDADeviceIndex()
        {
            return cudaDeviceIndex;
        }

        cudaStream_t getCUDAStream()
        {
            return cudaStream;
        }

        std::string getName() const override
        {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, cudaDeviceIndex);
            return prop.name;
        }

        std::string getInfo() const override
        {
            std::ostringstream result;
            cudaDeviceProp     prop;
            cudaGetDeviceProperties(&prop, cudaDeviceIndex);

            result << prop.name << " (" << prop.major << "." << prop.minor << ")\n";
            result << "\tType: " << (prop.integrated ? "Integrated" : "Discrete") << '\n';
            result << "\tCompute Units: " << prop.multiProcessorCount << '\n';
            result << "\tGlobal Memory Size: " << (prop.totalGlobalMem / 1000000) << " MB\n";
            result << "\tMaximum Object Size: " << (prop.maxMemoryAllocationSize / 1000000) << " MB\n";

            return result.str();
        }

    private:
        int cudaDeviceIndex;
        cudaStream_t cudaStream;
        bool initialized = false;
    };

} // namespace cle

#endif // __INCLUDE_DEVICE_HPP
