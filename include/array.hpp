#ifndef __INCLUDE_ARRAY_HPP
#define __INCLUDE_ARRAY_HPP

#include "backend.hpp"
#include "device.hpp"
#include "utils.hpp"

#include <algorithm>
#include <variant>

namespace cle
{

// @StRigaud TODO:
// - enable cl_image and cudaArray
// - enable backend management of cl_image and cudaArray
// - add memory type enum and friend operator (buffer, image)
// - add tests corresponding to cl_image and cudaArray managment
class Array
{
public:
  using Pointer = const Array *;

  [[nodiscard]] inline auto
  ptr() const -> Pointer
  {
    // @StRigaud: possible safety issue, using shared_ptr is pain ...
    // class Array : public std::enable_shared_from_this<Array> {
    // public:
    //     using Pointer = std::shared_ptr<const Array>;
    //     static auto create() -> Pointer {
    //         return std::make_shared<Array>();
    //     }
    //}
    // Array::Pointer arr = Array::create();
    // Array::Pointer arr_ptr = arr->ptr();
    return this;
  }

  Array() = default;
  Array(const size_t & width,
        const size_t & height,
        const size_t & depth,
        const dType &  data_type,
        const mType &  mem_type);
  Array(const size_t &          width,
        const size_t &          height,
        const size_t &          depth,
        const dType &           data_type,
        const mType &           mem_type,
        const Device::Pointer & device_ptr);
  Array(const size_t &          width,
        const size_t &          height,
        const size_t &          depth,
        const dType &           data_type,
        const mType &           mem_type,
        const void *            host_data,
        const Device::Pointer & device_ptr);
  Array(const Array & arr);
  ~Array();

  auto
  allocate() -> void;
  auto
  write(const void * host_data) -> void;
  auto
  read(void * host_data) const -> void;
  auto
  copy(const Array & dst) const -> void;

  template <typename T>
  auto
  fill(const T & value) const -> void
  {
    if (!initialized())
    {
      std::cerr << "Error: Arrays are not initialized_" << std::endl;
    }
    if (dim() > 1)
    {
      backend_.setMemory(
        device(), get(), width(), height(), depth(), bytesPerElements(), static_cast<const void *>(&value));
    }
    else
    {
      backend_.setMemory(
        device(), get(), nbElements() * bytesPerElements(), static_cast<const void *>(&value), bytesPerElements());
    }
    // backend_.setMemory(device(), get(), nbElements() * bytesPerElements(), static_cast<const void *>(&value),
    // sizeof(T));
  }

  [[nodiscard]] auto
  nbElements() const -> size_t;
  [[nodiscard]] auto
  width() const -> size_t;
  [[nodiscard]] auto
  height() const -> size_t;
  [[nodiscard]] auto
  depth() const -> size_t;
  [[nodiscard]] auto
  bytesPerElements() const -> size_t;
  [[nodiscard]] auto
  dtype() const -> dType;
  [[nodiscard]] auto
  mtype() const -> mType;
  [[nodiscard]] auto
  device() const -> Device::Pointer;
  [[nodiscard]] auto
  dim() const -> unsigned int;
  [[nodiscard]] auto
  initialized() const -> bool;
  [[nodiscard]] auto
  shortType() const -> std::string;
  [[nodiscard]] auto
  get() const -> void **;
  [[nodiscard]] auto
  c_get() const -> const void **;

  friend auto
  operator<<(std::ostream & out, const Array & array) -> std::ostream &
  {
    out << array.memType_ << " Array ([" << array.width_ << "," << array.height_ << "," << array.depth_
        << "], dtype=" << array.dtype() << ")";
    return out;
  }

private:
  using MemoryPointer = std::shared_ptr<void *>;

  mType           memType_ = mType::Buffer;
  dType           dataType_ = dType::Float;
  size_t          width_ = 1;
  size_t          height_ = 1;
  size_t          depth_ = 1;
  bool            initialized_ = false;
  Device::Pointer device_ = nullptr;
  MemoryPointer   data_ = std::make_shared<void *>(nullptr);
  const Backend & backend_ = cle::BackendManager::getInstance().getBackend();
};

} // namespace cle

#endif // __INCLUDE_ARRAY_HPP
