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
        Device() = default;
        virtual ~Device() = default;

        virtual void initialize() = 0;
        virtual void finalize() = 0;

        virtual bool isInitialized() const = 0;
        virtual std::string getName() const = 0;
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

    private:
        int cudaDeviceIndex;
        cudaStream_t cudaStream;
        bool initialized = false;
    };

} // namespace cle

#endif // __INCLUDE_DEVICE_HPP
