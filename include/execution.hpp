#ifndef __INCLUDE_EXECUTION_HPP
#define __INCLUDE_EXECUTION_HPP

#include "array.hpp"
#include "backend.hpp"
#include "device.hpp"

#include <array>
#include <variant>

namespace cle
{

using DevicePtr = std::shared_ptr<cle::Device>;
using ParameterMap = std::vector<std::pair<std::string, std::variant<Array, float, int>>>;
using ConstantMap = std::map<std::string, int>;
using KernelInfo = std::pair<std::string, std::string>;
using RangeArray = std::array<size_t, 3>;

static auto
cudaDefines(const ParameterMap & parameter_list, const ConstantMap & constant_list) -> std::string;

static auto
oclDefines(const ParameterMap & parameter_list, const ConstantMap & constant_list) -> std::string;

auto
execute(const DevicePtr &    device,
        const KernelInfo &   kernel_func,
        const ParameterMap & parameters,
        const ConstantMap &  constants = {},
        const RangeArray &   global_range = { 1, 1, 1 }) -> void;

} // namespace cle

#endif // __INCLUDE_EXECUTION_HPP
