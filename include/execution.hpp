#ifndef __INCLUDE_EXECUTION_HPP
#define __INCLUDE_EXECUTION_HPP

#include "array.hpp"
#include "backend.hpp"
#include "device.hpp"

#include <variant>

namespace cle
{
    using DevicePtr = std::shared_ptr<cle::Device>;
    using ParameterMap = std::map<std::string, std::variant<Array, float, int>>;
    using KernelInfo = std::pair<std::string, std::string>;
    using RangeArray = std::array<size_t, 3>;

    static auto execute(const KernelInfo &kernel_func, const ParameterMap &parameters, const RangeArray &global_rage, const DevicePtr &device) -> void
    {
        std::vector<void *> args_ptr;
        std::vector<size_t> args_size;
        args_ptr.reserve(parameters.size());
        args_size.reserve(parameters.size());

        std::string func_name = kernel_func.first;
        std::string kernel_source = kernel_func.second;
        std::string preamble = cle::BackendManager::getInstance().getBackend().getPreamble();
        // TODO: build defines from parameters

        for (const auto &[key, value] : parameters)
        {
            if (std::holds_alternative<Array>(value))
            {
                const auto &arr = std::get<Array>(value);
                args_ptr.push_back(arr.get());
                args_size.push_back(arr.nbElements() * arr.bytesPerElement());
            }
            else if (std::holds_alternative<float>(value))
            {
                const auto &f = std::get<float>(value);
                args_ptr.push_back(const_cast<float *>(&f));
                args_size.push_back(sizeof(float));
            }
            else if (std::holds_alternative<int>(value))
            {
                const auto &i = std::get<int>(value);
                args_ptr.push_back(const_cast<int *>(&i));
                args_size.push_back(sizeof(int));
            }
        }

        cle::BackendManager::getInstance().getBackend().executeKernel(device, source, func_name, global_rage, args_ptr, args_size);
    }

} // namespace cle

#endif // __INCLUDE_EXECUTION_HPP
