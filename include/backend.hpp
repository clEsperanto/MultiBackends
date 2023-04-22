#ifndef __INCLUDE_BACKEND_HPP
#define __INCLUDE_BACKEND_HPP

#include <optional>

#include <CL/opencl.hpp>
#include <cuda_runtime.h>

#include "device.hpp"

namespace cle
{
    class Backend
    {
    public:
        // enum for backend types
        enum class Type
        {
            CUDA,
            OPENCL
        };

        Backend() = default;
        virtual ~Backend() = default;

        virtual Backend::Type getType() = 0;
        virtual std::vector<std::string> getDeviceList(const std::optional<std::string> &type) = 0;
        virtual std::vector<cle::Device> getDevices(const std::optional<std::string> &type) = 0;

        virtual void allocateMemory(const Device& device, const size_t& size, void* data_ptr) = 0;
        virtual void freeMemory(const Device& device, void* data_ptr) = 0;
        virtual void writeMemory(const Device& device, void* data_ptr, const size_t& size, const void* host_ptr) = 0;
        virtual void readMemory(const Device& device, const void* data_ptr, const size_t& size, void* host_ptr) = 0;
    };

    class CUDABackend : public Backend
    {
    public:
        CUDABackend() = default;
        virtual ~CUDABackend() = default;

        virtual std::vector<cle::CUDADevice> getDevices(const std::optional<std::string> &type = std::nullopt) override
        {
            int deviceCount;
            cudaError_t error = cudaGetDeviceCount(&deviceCount);

            std::vector<cle::Device> devices;
            for (int i = 0; i < deviceCount; i++)
            {
                devices.push_back(cle::CUDADevice(i));
            }

            return devices;
        }

        virtual std::vector<std::string> getDeviceList(const std::optional<std::string> &type = std::nullopt) override
        {
            auto devices = getDevices(type);

            std::vector<std::string> deviceList;
            for (int i = 0; i < devices.size(); i++)
            {
                deviceList.push_back(devices[i].getName());
            }
        }

        Backend::Type getType() override
        {
            return Backend::Type::CUDA;
        }

        void allocateMemory(const Device& device, const size_t& size, void* data_ptr) override
        {
            cudaError_t err = cudaSetDevice(device.getCUDeviceID());
            if (err != cudaSuccess) {
                throw std::runtime_error("Error: Failed to set CUDA device");
            }

            cudaError_t err = cudaMalloc(data_ptr, size);
            if (err != cudaSuccess) {
                throw std::runtime_error("Error: Failed to allocate CUDA memory");
            }
        }

        void freeMemory(const Device& device, void* data_ptr) override
        {
            cudaError_t err = cudaSetDevice(device.getCUDeviceID());
            if (err != cudaSuccess) {
                throw std::runtime_error("Error: Failed to set CUDA device");
            }

            cudaError_t err = cudaFree(data_ptr);
            if (err != cudaSuccess) {
                throw std::runtime_error("Error: Failed to free CUDA memory");
            }
        }

        void writeMemory(const Device& device, void* data_ptr, const size_t& size, const void* host_ptr) override
        {
            cudaError_t err = cudaSetDevice(device.getCUDeviceID());
            if (err != cudaSuccess) {
                throw std::runtime_error("Error: Failed to set CUDA device");
            }

            cudaError_t err = cudaMemcpy(data_ptr, host_ptr, size, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                throw std::runtime_error("Error: Failed to write CUDA memory");
            }
        }

        void readMemory(const Device& device, const void* data_ptr, const size_t& size, void* host_ptr) override
        {
            cudaError_t err = cudaSetDevice(device.getCUDeviceID());
            if (err != cudaSuccess) {
                throw std::runtime_error("Error: Failed to set CUDA device");
            }
            
            cudaError_t err = cudaMemcpy(host_ptr, data_ptr, size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess) {
                throw std::runtime_error("Error: Failed to read CUDA memory");
            }
        }

    };

    class OpenCLBackend : public Backend
    {
    public:
        OpenCLBackend() = default;
        virtual ~OpenCLBackend() = default;

        virtual std::vector<cle::OpenCLDevice> getDevices(const std::optional<std::string> &type = std::make_optional<std::string>("all")) override
        {
            // Get the OpenCL platforms and devices
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);

            std::vector<cle::Device> devices;

            cl_device_type deviceType;
            if (type.value() == "cpu")
            {
                deviceType = CL_DEVICE_TYPE_CPU;
            }
            else if (type.value() == "gpu")
            {
                deviceType = CL_DEVICE_TYPE_GPU;
            }
            else if (type.value() == "all")
            {
                deviceType = CL_DEVICE_TYPE_ALL;
            }
            else
            {
                std::cerr << "Warning: Unknown device type '" << type.value() << "' provided." << std::endl;
                std::cerr << "\tdefault: fetching all devices." << std::endl;
            }

            for (int i = 0; i < platforms.size(); i++)
            {
                std::vector<cl::Device> clDevices;
                platforms[i].getDevices(deviceType, &clDevices);

                for (int j = 0; j < clDevices.size(); j++)
                {
                    devices.push_back(cle::OpenCLDevice(clDevices[j]));
                }
            }

            return devices;
        }

        virtual std::vector<std::string> getDeviceList(const std::optional<std::string> &type = std::make_optional<std::string>("all")) override override
        {
            auto devices = getDevices(type);

            std::vector<std::string> deviceList;
            for (int i = 0; i < devices.size(); i++)
            {
                deviceList.push_back(devices[i].getName());
            }

            return deviceList;
        }

        virtual Backend::Type getType() override
        {
            return Backend::Type::OPENCL;
        }

        void allocateMemory(const Device& device, const size_t& size, void* data_ptr) override
        {
            cl_int err;
            cl::Memory* mem_ptr = reinterpret_cast<cl::Memory*>(data_ptr);
            *mem_ptr = cl::Buffer(device.getCLContext(), CL_MEM_READ_WRITE, size, nullptr, &err);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("Error: Failed to allocate OpenCL memory");
            }
        }

        void freeMemory(const Device& device, void* data_ptr) override
        {
            cl::Memory* mem_ptr = reinterpret_cast<cl::Memory*>(data_ptr);
            cl_int err = clReleaseMemObject(*mem_ptr);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("Error: Failed to free OpenCL memory");
            }
        }

        void writeMemory(const Device& device, void* data_ptr, const size_t& size, const void* host_ptr) override
        {
            cl_int err = device.getCLCommandQueue().enqueueWriteBuffer(*reinterpret_cast<cl::Buffer*>(data_ptr), CL_TRUE, 0, size, host_ptr);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("Error: Failed to write OpenCL memory");
            }
        }

        void readMemory(const Device& device, const void* data_ptr, const size_t& size, void* host_ptr) override
        {
            cl_int err = device.getCLCommandQueue().enqueueReadBuffer(*reinterpret_cast<cl::Buffer*>(data_ptr), CL_TRUE, 0, size, host_ptr);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("Error: Failed to read OpenCL memory");
            }
        }
    
    };

    class BackendManager 
    {
    public:
        static BackendManager& getInstance() {
            static BackendManager instance;
            return instance;
        }

        void selectBackend(bool useCUDA) {
            if (useCUDA) {
                backend = std::make_unique<CUDABackend>();
            } else {
                backend = std::make_unique<OpenCLBackend>();
            }
        }

        Backend& getBackend() {
            if (!backend) {
                throw std::runtime_error("Backend not selected.");
            }
            return *backend;
        }

    private:
        std::unique_ptr<Backend> backend;

        BackendManager() {}
        BackendManager(const BackendManager&) = delete;
        BackendManager& operator=(const BackendManager&) = delete;
    };


} // namespace cle

#endif // __INCLUDE_BACKEND_HPP
