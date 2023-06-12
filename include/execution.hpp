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
using ParameterList =
  std::vector<std::pair<std::string, std::variant<std::reference_wrapper<const Array>, const float, const int>>>;
using ConstantList = std::vector<std::pair<std::string, int>>;
using KernelInfo = std::pair<std::string, std::string>;
using RangeArray = std::array<size_t, 3>;

static auto
cudaDefines(const ParameterList & parameter_list, const ConstantList & constant_list) -> std::string;

static auto
oclDefines(const ParameterList & parameter_list, const ConstantList & constant_list) -> std::string;

auto
execute(const DevicePtr &     device,
        const KernelInfo &    kernel_func,
        const ParameterList & parameters,
        const ConstantList &  constants = {},
        const RangeArray &    global_range = { 1, 1, 1 }) -> void;

auto
loadSource(const std::string & source_path) -> std::string;

} // namespace cle

#endif // __INCLUDE_EXECUTION_HPP
