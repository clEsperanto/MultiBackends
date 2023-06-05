#ifndef __INCLUDE_MEMORY_HPP
#define __INCLUDE_MEMORY_HPP

#include "array.hpp"
// #include "backend.hpp"
#include "device.hpp"
#include "utils.hpp"

namespace cle::memory
{
using DevicePtr = std::shared_ptr<cle::Device>;

template <typename T>
static auto
create(const size_t & width, const size_t & height, const size_t & depth, const DevicePtr & device) -> Array
{
  return Array{ width, height, depth, toType<T>(), mType::Buffer, device };
}

static auto
create_like(const Array & arr) -> Array
{
  return Array{ arr.width(), arr.height(), arr.depth(), arr.dtype(), arr.mtype(), arr.device() };
}

template <typename T>
static auto
push(const T * host_data, const size_t & width, const size_t & height, const size_t & depth, const DevicePtr & device)
  -> Array
{
  return Array{ width, height, depth, toType<T>(), mType::Buffer, host_data, device };
}

template <typename T>
static auto
pull(const Array & arr, T * host_arr) -> void
{
  arr.read(host_arr);
}

template <typename T>
static auto
copy(const Array & src) -> Array
{
  Array dst{ src.width(), src.height(), src.depth(), toType<T>(), mType::Buffer, src.device() };
  src.copy(dst);
  return dst;
}

} // namespace cle::memory

#endif // __INCLUDE_MEMORY_HPP
