#ifndef __COBRA_ARRAY_HPP
#define __COBRA_ARRAY_HPP

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
  using DevicePtr = std::shared_ptr<cle::Device>;
  using DataPtr = std::shared_ptr<void *>;

  Array() = default;
  Array(const size_t & width,
        const size_t & height,
        const size_t & depth,
        const dType &  data_type,
        const mType &  mem_type)
    : width_(width)
    , height_(height)
    , depth_(depth)
    , dataType_(data_type)
    , memType_(mem_type)
  {
    width_ = (width_ > 1) ? width_ : 1;
    height_ = (height_ > 1) ? height_ : 1;
    depth_ = (depth_ > 1) ? depth_ : 1;
  }

  Array(const size_t &    width,
        const size_t &    height,
        const size_t &    depth,
        const dType &     data_type,
        const mType &     mem_type,
        const DevicePtr & device_ptr)
    : Array(width, height, depth, data_type, mem_type)
  {
    device_ = device_ptr;
    if (dim() > 1)
    {
      backend_.allocateMemory(device(), this->width(), this->height(), this->depth(), dType(), get());
    }
    else
    {
      backend_.allocateMemory(device(), nbElements() * bytesPerElements(), get());
    }
    initialized_ = true;
  }

  Array(const size_t &    width,
        const size_t &    height,
        const size_t &    depth,
        const dType &     data_type,
        const mType &     mem_type,
        const void *      host_data,
        const DevicePtr & device_ptr)
    : Array(width, height, depth, data_type, mem_type, device_ptr)
  {
    if (dim() > 1)
    {
      backend_.writeMemory(
        device(), get(), this->width(), this->height(), this->depth(), bytesPerElements(), host_data);
    }
    else
    {
      backend_.writeMemory(device(), get(), nbElements() * bytesPerElements(), host_data);
    }
  }

  ~Array()
  {
    if (initialized() && data_.unique())
    {
      backend_.freeMemory(device(), mtype(), get());
    }
  }

  auto
  allocate() -> void
  {
    if (!initialized())
    {
      backend_.allocateMemory(device(), nbElements() * bytesPerElements(), get());
      initialized_ = true;
    }
    else
    {
      std::cerr << "Warning: Array is already initialized_" << std::endl;
    }
  }

  auto
  write(const void * host_data) -> void
  {
    if (!initialized())
    {
      allocate();
    }
    if (dim() > 1)
    {
      backend_.writeMemory(
        device(), get(), this->width(), this->height(), this->depth(), bytesPerElements(), host_data);
    }
    else
    {
      backend_.writeMemory(device(), get(), nbElements() * bytesPerElements(), host_data);
    }
  }

  auto
  read(void * host_data) const -> void
  {
    if (!initialized())
    {
      throw std::runtime_error("Error: Array is not initialized, it cannot be read");
    }
    if (dim() > 1)
    {
      backend_.readMemory(device(), c_get(), width(), height(), depth(), bytesPerElements(), host_data);
    }
    else
    {
      backend_.readMemory(device(), c_get(), nbElements() * bytesPerElements(), host_data);
    }
  }

  auto
  copy(const Array & dst) const -> void
  {
    if (!initialized() || !dst.initialized())
    {
      std::cerr << "Error: Arrays are not initialized_" << std::endl;
    }
    if (device() != dst.device())
    {
      std::cerr << "Error: copying Arrays from different devices" << std::endl;
    }
    if (width() != dst.width() || height() != dst.height() || depth() != dst.depth() ||
        bytesPerElements() != dst.bytesPerElements())
    {
      std::cerr << "Error: Arrays dimensions do not match" << std::endl;
    }
    if (dim() > 1)
    {
      backend_.copyMemory(device(), c_get(), width(), height(), depth(), bytesPerElements(), dst.get());
    }
    else
    {
      backend_.copyMemory(device(), c_get(), nbElements() * bytesPerElements(), dst.get());
    }
  }

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
      backend_.setMemory(device(), get(), width(), height(), depth(), bytesPerElements(), (const void *)&value);
    }
    else
    {
      backend_.setMemory(device(), get(), nbElements() * bytesPerElements(), (const void *)&value, bytesPerElements());
    }
    // backend_.setMemory(device(), get(), nbElements() * bytesPerElements(), (const void *)&value, sizeof(T));
  }

  [[nodiscard]] auto
  nbElements() const -> size_t
  {
    return width_ * height_ * depth_;
  }
  [[nodiscard]] auto
  width() const -> size_t
  {
    return width_;
  }
  [[nodiscard]] auto
  height() const -> size_t
  {
    return height_;
  }
  [[nodiscard]] auto
  depth() const -> size_t
  {
    return depth_;
  }
  [[nodiscard]] auto
  bytesPerElements() const -> size_t
  {
    return toBytes(dataType_);
  }
  [[nodiscard]] auto
  dtype() const -> dType
  {
    return dataType_;
  }
  [[nodiscard]] auto
  mtype() const -> mType
  {
    return memType_;
  }
  [[nodiscard]] auto
  device() const -> DevicePtr
  {
    return device_;
  }
  [[nodiscard]] auto
  dim() const -> unsigned int
  {
    return (depth_ > 1) ? 3 : (height_ > 1) ? 2 : 1;
  }
  [[nodiscard]] auto
  initialized() const -> bool
  {
    return initialized_;
  }

  friend auto
  operator<<(std::ostream & out, const Array & array) -> std::ostream &
  {
    out << array.memType_ << " Array ([" << array.width_ << "," << array.height_ << "," << array.depth_
        << "], dtype=" << array.bytesPerElements() << ")";
    return out;
  }

  auto
  shortType() const -> std::string
  {
    switch (this->dataType_)
    {
      case dType::Float:
        return "f";
      case dType::Int32:
        return "i";
      case dType::UInt32:
        return "ui";
      case dType::Int8:
        return "c";
      case dType::UInt8:
        return "uc";
      case dType::Int16:
        return "s";
      case dType::UInt16:
        return "us";
      case dType::Int64:
        return "l";
      case dType::UInt64:
        return "ul";
      default:
        throw std::invalid_argument("Invalid Array::Type value");
    }
  }

  [[nodiscard]] auto
  get() const -> void **
  {
    return data_.get();
  }
  [[nodiscard]] auto
  c_get() const -> const void **
  {
    return (const void **)data_.get();
  }

private:
  mType           memType_ = mType::Buffer;
  dType           dataType_ = dType::Float;
  size_t          width_ = 1;
  size_t          height_ = 1;
  size_t          depth_ = 1;
  bool            initialized_ = false;
  DevicePtr       device_ = nullptr;
  DataPtr         data_ = std::make_shared<void *>(nullptr);
  const Backend & backend_ = cle::BackendManager::getInstance().getBackend();
};

} // namespace cle

#endif // __COBRA_ARRAY_HPP
