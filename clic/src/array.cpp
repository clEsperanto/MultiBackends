#include "array.hpp"

namespace cle
{

Array::Array(const size_t &          width,
             const size_t &          height,
             const size_t &          depth,
             const dType &           data_type,
             const mType &           mem_type,
             const Device::Pointer & device_ptr)
  : width_(width)
  , height_(height)
  , depth_(depth)
  , dataType_(data_type)
  , memType_(mem_type)
  , device_(device_ptr)
  , data_(std::make_shared<void *>(nullptr))
  , initialized_(false)
{
  width_ = (width_ > 1) ? width_ : 1;
  height_ = (height_ > 1) ? height_ : 1;
  depth_ = (depth_ > 1) ? depth_ : 1;
}

Array::~Array()
{
  if (initialized() && data_.unique())
  {
    backend_.freeMemory(device(), mtype(), get());
  }
}

auto
Array::create(const size_t &          width,
              const size_t &          height,
              const size_t &          depth,
              const dType &           data_type,
              const mType &           mem_type,
              const Device::Pointer & device_ptr) -> Array::Pointer
{
  auto ptr = std::shared_ptr<Array>(new Array(width, height, depth, data_type, mem_type, device_ptr));
  ptr->allocate();
  return ptr;
}

auto
Array::create(const size_t &          width,
              const size_t &          height,
              const size_t &          depth,
              const dType &           data_type,
              const mType &           mem_type,
              const void *            host_data,
              const Device::Pointer & device_ptr) -> Array::Pointer
{
  auto ptr = create(width, height, depth, data_type, mem_type, device_ptr);
  ptr->write(host_data);
  return ptr;
}

auto
Array::create(Array::Pointer array) -> Array::Pointer
{
  auto ptr = create(array->width(), array->height(), array->depth(), array->dtype(), array->mtype(), array->device());
  array->copy(ptr);
  return ptr;
}

auto
operator<<(std::ostream & out, const Array::Pointer & array) -> std::ostream &
{
  out << "Array ([" << array->width() << "," << array->height() << "," << array->depth()
      << "], dtype=" << array->dtype() << ", mtype=" << array->mtype() << ")";
  return out;
}

auto
Array::allocate() -> void
{
  if (initialized())
  {
    std::cerr << "Warning: Array is already initialized" << std::endl;
    return;
  }
  if (mtype() == mType::Image)
  {
    backend_.allocateMemory(device(), this->width(), this->height(), this->depth(), dtype(), get());
  }
  else
  {
    backend_.allocateMemory(device(), nbElements() * bytesPerElements(), get());
  }
  initialized_ = true;
}

auto
Array::write(const void * host_data) -> void
{
  if (!initialized())
  {
    allocate();
  }
  if (mtype() == mType::Image)
  {
    backend_.writeMemory(device(), get(), this->width(), this->height(), this->depth(), bytesPerElements(), host_data);
  }
  else
  {
    backend_.writeMemory(device(), get(), nbElements() * bytesPerElements(), host_data);
  }
}

auto
Array::read(void * host_data) const -> void
{
  if (!initialized())
  {
    throw std::runtime_error("Error: Array is not initialized, it cannot be read");
  }
  if (mtype() == mType::Image)
  {
    backend_.readMemory(device(), c_get(), width(), height(), depth(), bytesPerElements(), host_data);
  }
  else
  {
    backend_.readMemory(device(), c_get(), nbElements() * bytesPerElements(), host_data);
  }
}

auto
Array::copy(const Array::Pointer & dst) const -> void
{
  if (!initialized() || !dst->initialized())
  {
    std::cerr << "Error: Arrays are not initialized_" << std::endl;
  }
  if (device() != dst->device())
  {
    std::cerr << "Error: copying Arrays from different devices" << std::endl;
  }
  if (width() != dst->width() || height() != dst->height() || depth() != dst->depth() ||
      bytesPerElements() != dst->bytesPerElements())
  {
    std::cerr << "Error: Arrays dimensions do not match" << std::endl;
  }
  if (mtype() == mType::Buffer && dst->mtype() == mType::Buffer)
  {
    backend_.copyMemoryBufferToBuffer(device(), c_get(), nbElements() * bytesPerElements(), dst->get());
  }
  else if (mtype() == mType::Image && dst->mtype() == mType::Image)
  {
    backend_.copyMemoryImageToImage(device(), c_get(), width(), height(), depth(), toBytes(dtype()), dst->get());
  }
  else if (mtype() == mType::Buffer && dst->mtype() == mType::Image)
  {
    backend_.copyMemoryBufferToImage(
      device(), c_get(), dst->width(), dst->height(), dst->depth(), toBytes(dst->dtype()), dst->get());
  }
  else if (mtype() == mType::Image && dst->mtype() == mType::Buffer)
  {
    backend_.copyMemoryImageToBuffer(device(), c_get(), width(), height(), depth(), toBytes(dtype()), dst->get());
  }
  else
  {
    std::cerr << "Error: copying Arrays from different memory types" << std::endl;
  }
}

auto
Array::fill(const float & value) const -> void
{
  if (!initialized())
  {
    std::cerr << "Error: Arrays are not initialized_" << std::endl;
  }
  if (mtype() == mType::Image)
  {
    backend_.setMemory(device(), get(), width(), height(), depth(), value, dtype());
  }
  else
  {
    backend_.setMemory(device(), get(), nbElements() * bytesPerElements(), value, dtype());
  }
}

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
Array::get() const -> void **
{
  return data_.get();
}
auto
Array::c_get() const -> const void **
{
  return (const void **)data_.get();
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

} // namespace cle
