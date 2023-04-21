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
        Backend() = default;
        virtual ~Backend() = default;

        virtual std::vector<std::string> getDeviceList(const std::optional<std::string> &type) = 0;
        virtual std::vector<cle::Device> getDevices(const std::optional<std::string> &type) = 0;
    };

    class CUDABackend : public Backend
    {
    public:
        CUDABackend() = default;
        virtual ~CUDABackend() = default;

        virtual std::vector<cle::Device> getDevices(const std::optional<std::string> &type = std::nullopt) override
        {
            int deviceCount;
            cudaError_t error = cudaGetDeviceCount(&deviceCount);

            std::vector<cle::Device> devices;
            for (int i = 0; i < deviceCount; i++)
            {
                devices.push_back(cle::Device(i));
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
    };

    class OpenCLBackend : public Backend
    {
    public:
        OpenCLBackend() = default;
        virtual ~OpenCLBackend() = default;

        virtual std::vector<cle::Device> getDevices(const std::optional<std::string> &type = std::make_optional<std::string>("all")) override
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
                    devices.push_back(cle::Device(clDevices[j]));
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
    };

} // namespace cle

#endif // __INCLUDE_BACKEND_HPP
