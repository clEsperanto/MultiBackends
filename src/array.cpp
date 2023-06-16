#include "array.hpp"

namespace cle
{

Array::Array(const size_t & width,
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

Array::Array(const size_t &          width,
             const size_t &          height,
             const size_t &          depth,
             const dType &           data_type,
             const mType &           mem_type,
             const Device::Pointer & device_ptr)
  : Array(width, height, depth, data_type, mem_type)
{
  device_ = device_ptr;
  allocate();
  // if (dim() > 1)
  // {
  //   backend_.allocateMemory(device(), this->width(), this->height(), this->depth(), dType(), get());
  // }
  // else
  // {
  //   backend_.allocateMemory(device(), nbElements() * bytesPerElements(), get());
  // }
  // initialized_ = true;
}

Array::Array(const size_t &          width,
             const size_t &          height,
             const size_t &          depth,
             const dType &           data_type,
             const mType &           mem_type,
             const void *            host_data,
             const Device::Pointer & device_ptr)
  : Array(width, height, depth, data_type, mem_type, device_ptr)
{
  // if (dim() > 1)
  // {
  //   backend_.writeMemory(device(), get(), this->width(), this->height(), this->depth(), bytesPerElements(),
  //   host_data);
  // }
  // else
  // {
  backend_.writeMemory(device(), get(), nbElements() * bytesPerElements(), host_data);
  // }
}

Array::Array(const Array & arr)
  : Array(arr.width(), arr.height(), arr.depth(), arr.dtype(), arr.mtype())
{
  device_ = arr.device();
  allocate();
}

Array::~Array()
{
  if (initialized() && data_.unique())
  {
    backend_.freeMemory(device(), mtype(), get());
  }
}

auto
Array::allocate() -> void
{
  if (!initialized())
  {
    // backend_.allocateMemory(device(), nbElements() * bytesPerElements(), get());
    // initialized_ = true;
    // if (dim() > 1)
    // {
    //   backend_.allocateMemory(device(), this->width(), this->height(), this->depth(), dType(), get());
    // }
    // else
    // {
    backend_.allocateMemory(device(), nbElements() * bytesPerElements(), get());
    // }
    initialized_ = true;
  }
  else
  {
    std::cerr << "Warning: Array is already initialized_" << std::endl;
  }
}

auto
Array::write(const void * host_data) -> void
{
  if (!initialized())
  {
    allocate();
  }
  // if (dim() > 1)
  // {
  //   backend_.writeMemory(device(), get(), this->width(), this->height(), this->depth(), bytesPerElements(),
  //   host_data);
  // }
  // else
  // {
  backend_.writeMemory(device(), get(), nbElements() * bytesPerElements(), host_data);
  // }
}

auto
Array::read(void * host_data) const -> void
{
  if (!initialized())
  {
    throw std::runtime_error("Error: Array is not initialized, it cannot be read");
  }
  // if (dim() > 1)
  // {
  //   backend_.readMemory(device(), c_get(), width(), height(), depth(), bytesPerElements(), host_data);
  // }
  // else
  // {
  backend_.readMemory(device(), c_get(), nbElements() * bytesPerElements(), host_data);
  // }
}

auto
Array::copy(const Array & dst) const -> void
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
  // if (dim() > 1)
  // {
  //   backend_.copyMemory(device(), c_get(), width(), height(), depth(), bytesPerElements(), dst.get());
  // }
  // else
  // {
  backend_.copyMemory(device(), c_get(), nbElements() * bytesPerElements(), dst.get());
  // }
}

// template <typename T>
// auto
// Array::fill(const T & value) const -> void
// {
//   if (!initialized())
//   {
//     std::cerr << "Error: Arrays are not initialized_" << std::endl;
//   }
//   if (dim() > 1)
//   {
//     backend_.setMemory(device(), get(), width(), height(), depth(), bytesPerElements(), (const void *)&value);
//   }
//   else
//   {
//     backend_.setMemory(device(), get(), nbElements() * bytesPerElements(), (const void *)&value, bytesPerElements());
//   }
//   // backend_.setMemory(device(), get(), nbElements() * bytesPerElements(), (const void *)&value, sizeof(T));
// }

auto
Array::nbElements() const -> size_t
{
  return width_ * height_ * depth_;
}
auto
Array::width() const -> size_t
{
  return width_;
}
auto
Array::height() const -> size_t
{
  return height_;
}
auto
Array::depth() const -> size_t
{
  return depth_;
}
auto
Array::bytesPerElements() const -> size_t
{
  return toBytes(dataType_);
}
auto
Array::dtype() const -> dType
{
  return dataType_;
}
auto
Array::mtype() const -> mType
{
  return memType_;
}
auto
Array::device() const -> Device::Pointer
{
  return device_;
}
auto
Array::dim() const -> unsigned int
{
  return (depth_ > 1) ? 3 : (height_ > 1) ? 2 : 1;
}
auto
Array::initialized() const -> bool
{
  return initialized_;
}

auto
Array::shortType() const -> std::string
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

auto
Array::get() const -> void **
{
  return data_.get();
}

auto
Array::c_get() const -> const void **
{
  return (const void **)data_.get();
}

} // namespace cle
