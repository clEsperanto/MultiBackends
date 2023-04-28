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
#include <map>

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

        [[nodiscard]] virtual inline auto getType() const -> Backend::Type = 0;
        [[nodiscard]] virtual inline auto getDevicesList(const std::string &type) const -> std::vector<std::string> = 0;
        [[nodiscard]] virtual inline auto getDevices(const std::string &type) const -> std::vector<DevicePtr> = 0;
        [[nodiscard]] virtual inline auto getDevice(const std::string &name, const std::string &type) const -> DevicePtr = 0;

        virtual inline auto allocateMemory(const DevicePtr &device, const size_t &size, void **data_ptr) const -> void = 0;
        virtual inline auto freeMemory(const DevicePtr &device, void **data_ptr) const -> void = 0;
        virtual inline auto writeMemory(const DevicePtr &device, void **data_ptr, const size_t &size, const void *host_ptr) const -> void = 0;
        virtual inline auto readMemory(const DevicePtr &device, const void **data_ptr, const size_t &size, void *host_ptr) const -> void = 0;
        virtual inline auto copyMemory(const DevicePtr &device, const void **src_data_ptr, const size_t &size, void **dst_data_ptr) const -> void = 0;
        virtual inline auto setMemory(const DevicePtr &device, void **data_ptr, const size_t &size, const void *value, const size_t &value_size) const -> void = 0;

        virtual inline auto buildKernel(const DevicePtr &device, const std::string &kernel_source, const std::string &kernel_name, void *kernel) const -> void = 0;
        virtual inline auto loadProgramFromCache(const DevicePtr &device, const std::string &hash, void *program) const -> void = 0;
        virtual inline auto saveProgramToCache(const DevicePtr &device, const std::string &hash, void *program) const -> void = 0;

        // auto executeKernel(const DevicePtr &device, const std::string &kernel_source, const std::string &kernel_name, const std::array<size_t, 3> &global_size, const std::vector<void *> &args, const std::vector<void *> &sizes) const -> void {}
        // virtual inline auto getPreamble() const -> const std::string &;

        friend auto
        operator<<(std::ostream &out, const Backend::Type &backend_type) -> std::ostream &
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

        friend auto operator<<(std::ostream &out, const Backend &backend) -> std::ostream &
        {
            out << backend.getType() << " backend";
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

        [[nodiscard]] inline auto getDevices(const std::string &type) const -> std::vector<DevicePtr> override
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
            throw std::runtime_error("Error: CUDA backend is not enabled");
#endif
        }

        [[nodiscard]] inline auto getDevice(const std::string &name, const std::string &type) const -> DevicePtr override
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
            throw std::runtime_error("Error: CUDA backend is not enabled");
#endif
        }

        [[nodiscard]] inline auto getDevicesList(const std::string &type) const -> std::vector<std::string> override
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
            throw std::runtime_error("Error: CUDA backend is not enabled");
#endif
        }

        [[nodiscard]] inline auto getType() const -> Backend::Type override
        {
            return Backend::Type::CUDA;
        }

        inline auto allocateMemory(const DevicePtr &device, const size_t &size, void **data_ptr) const -> void override
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
            throw std::runtime_error("Error: CUDA backend is not enabled");
#endif
        }

        inline auto freeMemory(const DevicePtr &device, void **data_ptr) const -> void override
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
            throw std::runtime_error("Error: CUDA backend is not enabled");
#endif
        }

        inline auto writeMemory(const DevicePtr &device, void **data_ptr, const size_t &size, const void *host_ptr) const -> void override
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
            throw std::runtime_error("Error: CUDA backend is not enabled");
#endif
        }

        inline auto readMemory(const DevicePtr &device, const void **data_ptr, const size_t &size, void *host_ptr) const -> void override
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
            throw std::runtime_error("Error: CUDA backend is not enabled");
#endif
        }

        inline auto copyMemory(const DevicePtr &device, const void **src_data_ptr, const size_t &size, void **dst_data_ptr) const -> void override
        {
#if CLE_CUDA
            auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
            auto err = cudaSetDevice(cuda_device->getCUDADeviceIndex());
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to set CUDA device before memory allocation.");
            }
            err = cudaMemcpy(*dst_data_ptr, src_data_ptr, size, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to write CUDA memory.");
            }
#else
            throw std::runtime_error("Error: CUDA backend is not enabled");
#endif
        }

        inline auto setMemory(const DevicePtr &device, void **data_ptr, const size_t &size, const void *value, const size_t &value_size) const -> void override
        {
#if CLE_CUDA
            auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
            auto err = cudaSetDevice(cuda_device->getCUDADeviceIndex());
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to set CUDA device before memory allocation.");
            }
            err = cudaMemset(*data_ptr, *static_cast<const int *>(value), size);
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to set CUDA memory.");
            }
#else
            throw std::runtime_error("Error: CUDA backend is not enabled");
#endif
        }

        inline auto loadProgramFromCache(const DevicePtr &device, const std::string &hash, void *program) const -> void override
        {
#if CLE_CUDA
            auto cuda_device = std::dynamic_pointer_cast<CUDADevice>(device);
            CUmodule module = nullptr;
            auto ite = cuda_device->getCache().find(hash);
            if (ite != cuda_device->getCache().end())
            {
                module = ite->second;
            }
            program = module;
#else
            throw std::runtime_error("Error: CUDA backend is not enabled");
#endif
        }

        inline auto saveProgramToCache(const DevicePtr &device, const std::string &hash, void *program) const -> void override
        {
#if CLE_CUDA
            auto cuda_device = std::dynamic_pointer_cast<CUDADevice>(device);
            cuda_device->getCache().emplace_hint(cuda_device->getCache().end(), hash, (CUmodule)program);
#else
            throw std::runtime_error("Error: CUDA backend is not enabled");
#endif
        }

        inline auto buildKernel(const DevicePtr &device, const std::string &kernel_source, const std::string &kernel_name, void *kernel) const -> void override
        {
#if CLE_CUDA
            auto cuda_device = std::dynamic_pointer_cast<const CUDADevice>(device);
            auto err = cudaSetDevice(cuda_device->getCUDADeviceIndex());
            if (err != cudaSuccess)
            {
                throw std::runtime_error("Error: Failed to set CUDA device before memory allocation.");
            }
            CUmodule module;
            CUfunction function;
            std::string hash = std::to_string(std::hash<std::string>{}(kernel_source));
            loadProgramFromCache(device, hash, module);
            if (module == nullptr)
            {
                auto res = cuModuleLoadDataEx(&module, kernel_source.c_str(), 0, 0, 0);
                if (res != CUDA_SUCCESS)
                {
                    throw std::runtime_error("Error: Failed to build CUDA program.");
                }
                saveProgramToCache(device, hash, module);
            }
            auto res = cuModuleGetFunction(&function, module, kernel_name.c_str());
            if (res != CUDA_SUCCESS)
            {
                throw std::runtime_error("Error: Failed to get CUDA kernel.");
            }
            *((CUfunction *)kernel) = function;
#else
            throw std::runtime_error("Error: CUDA backend is not enabled");
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

        [[nodiscard]] inline auto getDevices(const std::string &type) const -> std::vector<DevicePtr> override
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

        [[nodiscard]] inline auto getDevice(const std::string &name, const std::string &type) const -> DevicePtr override
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

        [[nodiscard]] inline auto getDevicesList(const std::string &type) const -> std::vector<std::string> override
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

        [[nodiscard]] inline auto getType() const -> Backend::Type override
        {
            return Backend::Type::OPENCL;
        }

        inline auto allocateMemory(const DevicePtr &device, const size_t &size, void **data_ptr) const -> void override
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

        inline auto freeMemory(const DevicePtr &device, void **data_ptr) const -> void override
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

        inline auto writeMemory(const DevicePtr &device, void **data_ptr, const size_t &size, const void *host_ptr) const -> void override
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

        inline auto readMemory(const DevicePtr &device, const void **data_ptr, const size_t &size, void *host_ptr) const -> void override
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

        inline auto copyMemory(const DevicePtr &device, const void **src_data_ptr, const size_t &size, void **dst_data_ptr) const -> void override
        {
#if CLE_OPENCL
            auto opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);
            auto queue = opencl_device->getCLCommandQueue();
            auto err = clEnqueueCopyBuffer(queue.get(), *static_cast<const cl_mem *>(*src_data_ptr), *static_cast<cl_mem *>(*dst_data_ptr), 0, 0, size, 0, nullptr, nullptr);
            if (err != CL_SUCCESS)
            {
                throw std::runtime_error("Error: Failed to write OpenCL memory.");
            }
#else
            throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
        }

        inline auto setMemory(const DevicePtr &device, void **data_ptr, const size_t &size, const void *value, const size_t &value_size) const -> void override
        {
#if CLE_OPENCL
            auto opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);
            auto queue = opencl_device->getCLCommandQueue();
            auto err = clEnqueueFillBuffer(queue.get(), *static_cast<cl_mem *>(*data_ptr), value, value_size, 0, size, 0, nullptr, nullptr);
            if (err != CL_SUCCESS)
            {
                throw std::runtime_error("Error: Failed to set OpenCL memory.");
            }
#else
            throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
        }

        inline auto loadProgramFromCache(const DevicePtr &device, const std::string &hash, void *program) const -> void override
        {
#if CLE_OPENCL
            auto opencl_device = std::dynamic_pointer_cast<OpenCLDevice>(device);

            cl_program prog = nullptr;
            auto ite = opencl_device->getCache().find(hash);
            if (ite != opencl_device->getCache().end())
            {
                prog = ite->second;
            }
            *static_cast<cl_program *>(program) = prog;
#else
            throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
        }

        inline auto saveProgramToCache(const DevicePtr &device, const std::string &hash, void *program) const -> void override
        {

#if CLE_OPENCL
            auto opencl_device = std::dynamic_pointer_cast<OpenCLDevice>(device);
            opencl_device->getCache().emplace_hint(opencl_device->getCache().end(), hash, *static_cast<cl_program *>(program));
#else
            throw std::runtime_error("OpenCLBackend::getDevices: OpenCL is not enabled");
#endif
        }

        inline auto buildKernel(const DevicePtr &device, const std::string &kernel_source, const std::string &kernel_name, void *kernel) const -> void override
        {
#if CLE_OPENCL
            cl_int err;
            auto opencl_device = std::dynamic_pointer_cast<const OpenCLDevice>(device);
            auto context = opencl_device->getCLContext();
            cl_program prog = nullptr;
            std::string hash = std::to_string(std::hash<std::string>{}(kernel_source));
            loadProgramFromCache(device, hash, prog);
            if (prog == nullptr)
            {
                prog = clCreateProgramWithSource(context.get(), 1, (const char **)&kernel_source, nullptr, &err);
                if (err != CL_SUCCESS)
                {
                    size_t len;
                    char buffer[2048];
                    clGetProgramBuildInfo(prog, opencl_device->getCLDevice().get(), CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
                    std::cerr << buffer << std::endl;
                    throw std::runtime_error("Error: Failed to build OpenCL program.");
                }
                saveProgramToCache(device, hash, prog);
            }
            auto clkernel = clCreateKernel(prog, kernel_name.c_str(), nullptr);
            *static_cast<cl_kernel *>(kernel) = clkernel;
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

        inline auto setBackend(bool useCUDA) -> void
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

        [[nodiscard]] inline auto getBackend() const -> const Backend &
        {
            if (!this->backend)
            {
                throw std::runtime_error("Backend not selected.");
            }
            return *this->backend;
        }

        friend auto operator<<(std::ostream &out, const BackendManager &backend_manager) -> std::ostream &
        {
            out << backend_manager.getBackend().getType() << " backend";
            return out;
        }

        auto operator=(const BackendManager &) -> BackendManager & = delete;
        BackendManager(const BackendManager &) = delete;

    private:
        std::shared_ptr<Backend> backend;
        BackendManager() = default;
    };

} // namespace cle

#endif // __INCLUDE_BACKEND_HPP
