#ifndef __INCLUDE_ARRAY_HPP
#define __INCLUDE_ARRAY_HPP

#include "backend.hpp"
#include "device.hpp"

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

        enum class Type
        {
            Float,
            Int,
            UnsignedInt,
            Char,
            UnsignedChar,
            Short,
            UnsignedShort,
            Long,
            UnsignedLong
        };

        enum class Memory
        {
            Buffer,
            Image
        };

        Array() = default;
        Array(const size_t &width, const size_t &height, const size_t &depth, const Array::Type &data_type) : width_(width), height_(height), depth_(depth), dataType(data_type)
        {
            this->bytes_per_element_ = toBytes(dataType);
            this->width_ = (this->width_ > 1) ? this->width_ : 1;
            this->height_ = (this->height_ > 1) ? this->height_ : 1;
            this->depth_ = (this->depth_ > 1) ? this->depth_ : 1;
        }

        Array(const size_t &width, const size_t &height, const size_t &depth, const Array::Type &data_type, const DevicePtr &device) : Array(width, height, depth, data_type)
        {
            this->device_ = device;
            backend.allocateMemory(device_, this->nbElements() * bytes_per_element_, this->get());
            this->initialized = true;
        }

        Array(const size_t &width, const size_t &height, const size_t &depth, const Array::Type &data_type, const void *host_data, const DevicePtr &device) : Array(width, height, depth, data_type, device)
        {
            backend.writeMemory(device_, this->get(), this->nbElements() * bytes_per_element_, host_data);
        }

        ~Array()
        {
            if (this->initialized && data.unique())
            {
                backend.freeMemory(device_, this->get());
            }
        }

        auto allocate() -> void
        {
            if (!this->initialized)
            {
                backend.allocateMemory(device_, this->nbElements() * bytes_per_element_, this->get());
                this->initialized = true;
            }
            else
            {
                std::cerr << "Warning: Array is already initialized" << std::endl;
            }
        }

        auto write(const void *host_data) -> void
        {
            if (!this->initialized)
            {
                allocate();
            }
            backend.writeMemory(device_, this->get(), this->nbElements() * bytes_per_element_, host_data);
        }

        auto read(void *host_data) const -> void
        {
            backend.readMemory(device_, this->c_get(), this->nbElements() * bytes_per_element_, host_data);
        }

        auto copy(const Array &dst) const -> void
        {
            if (!this->initialized || !dst.initialized)
            {
                std::cerr << "Error: Arrays are not initialized" << std::endl;
            }
            if (this->device_ != dst.device_)
            {
                std::cerr << "Error: copying Arrays from different devices" << std::endl;
            }
            if (this->width_ != dst.width_ || this->height_ != dst.height_ || this->depth_ != dst.depth_ || this->bytes_per_element_ != dst.bytes_per_element_)
            {
                std::cerr << "Error: Arrays dimensions do not match" << std::endl;
            }
            backend.copyMemory(device_, this->c_get(), this->nbElements() * bytes_per_element_, dst.get());
        }

        template <typename T>
        auto fill(const T &value) const -> void
        {
            if (!this->initialized)
            {
                std::cerr << "Error: Arrays are not initialized" << std::endl;
            }
            backend.setMemory(device_, this->get(), this->nbElements() * this->bytes_per_element_, (const void *)&value, sizeof(T));
        }

        [[nodiscard]] auto
        nbElements() const -> const size_t
        {
            return width_ * height_ * depth_;
        }
        [[nodiscard]] auto width() const -> size_t { return width_; }
        [[nodiscard]] auto height() const -> size_t { return height_; }
        [[nodiscard]] auto depth() const -> size_t { return depth_; }
        [[nodiscard]] auto bytesPerElements() const -> size_t { return toBytes(dataType); }
        [[nodiscard]] auto dtype() const -> Array::Type { return dataType; }
        [[nodiscard]] auto device() const -> DevicePtr { return device_; }
        [[nodiscard]] auto dim() const -> unsigned int { return (depth_ > 1) ? 3 : (height_ > 1) ? 2
                                                                                                 : 1; }

        friend auto operator<<(std::ostream &out, const Array &array) -> std::ostream &
        {
            out << array.memType << " Array ([" << array.width_ << "," << array.height_ << "," << array.depth_ << "], dtype=" << array.bytes_per_element_ << ")";
            return out;
        }

        friend auto operator<<(std::ostream &out, const Array::Memory &mtype) -> std::ostream &
        {
            switch (mtype)
            {
            case Array::Memory::Buffer:
                out << "Buffer";
                break;
            case Array::Memory::Image:
                out << "Image";
                break;
            }
            return out;
        }

        friend auto operator<<(std::ostream &out, const Array::Type &dtype) -> std::ostream &
        {
            switch (dtype)
            {
            case Array::Type::Float:
                out << "float";
                break;
            case Array::Type::Int:
                out << "int";
                break;
            case Array::Type::UnsignedInt:
                out << "uint";
                break;
            case Array::Type::Char:
                out << "char";
                break;
            case Array::Type::UnsignedChar:
                out << "uchar";
                break;
            case Array::Type::Short:
                out << "short";
                break;
            case Array::Type::UnsignedShort:
                out << "ushort";
                break;
            case Array::Type::Long:
                out << "long";
                break;
            case Array::Type::UnsignedLong:
                out << "ulong";
                break;
            }
            return out;
        }

        friend auto toBytes(const Array::Type &dtype) -> const size_t
        {
            switch (dtype)
            {
            case Array::Type::Float:
                return sizeof(float);
            case Array::Type::Int:
                return sizeof(int32_t);
            case Array::Type::UnsignedInt:
                return sizeof(uint32_t);
            case Array::Type::Char:
                return sizeof(int8_t);
            case Array::Type::UnsignedChar:
                return sizeof(uint8_t);
            case Array::Type::Short:
                return sizeof(int16_t);
            case Array::Type::UnsignedShort:
                return sizeof(uint16_t);
            case Array::Type::Long:
                return sizeof(int64_t);
            case Array::Type::UnsignedLong:
                return sizeof(uint64_t);
            default:
                throw std::invalid_argument("Invalid Array::Type value");
            }
        }

        template <typename T>
        static auto toType() -> const Array::Type
        {
            if constexpr (std::is_same_v<T, float>)
            {
                return Array::Type::Float;
            }
            else if constexpr (std::is_same_v<T, int32_t>)
            {
                return Array::Type::Int;
            }
            else if constexpr (std::is_same_v<T, uint32_t>)
            {
                return Array::Type::UnsignedInt;
            }
            else if constexpr (std::is_same_v<T, int16_t>)
            {
                return Array::Type::Short;
            }
            else if constexpr (std::is_same_v<T, uint16_t>)
            {
                return Array::Type::UnsignedShort;
            }
            else if constexpr (std::is_same_v<T, int8_t>)
            {
                return Array::Type::Char;
            }
            else if constexpr (std::is_same_v<T, uint8_t>)
            {
                return Array::Type::UnsignedChar;
            }
            else if constexpr (std::is_same_v<T, int64_t>)
            {
                return Array::Type::Long;
            }
            else if constexpr (std::is_same_v<T, uint64_t>)
            {
                return Array::Type::UnsignedLong;
            }
            else
            {
                throw std::invalid_argument("Error: Invalid type");
            }
        }

        auto shortType() const -> const std::string
        {
            switch (dataType)
            {
            case Array::Type::Float:
                return "f";
            case Array::Type::Int:
                return "i";
            case Array::Type::UnsignedInt:
                return "ui";
            case Array::Type::Char:
                return "c";
            case Array::Type::UnsignedChar:
                return "uc";
            case Array::Type::Short:
                return "s";
            case Array::Type::UnsignedShort:
                return "us";
            case Array::Type::Long:
                return "l";
            case Array::Type::UnsignedLong:
                return "ul";
            default:
                throw std::invalid_argument("Invalid Array::Type value");
            }
        }

        [[nodiscard]] auto get() const -> void ** { return data.get(); }
        [[nodiscard]] auto c_get() const -> const void ** { return (const void **)data.get(); }

    private:
        Array::Memory memType = Array::Memory::Buffer;
        Array::Type dataType = Array::Type::Float;
        size_t bytes_per_element_ = sizeof(float);
        size_t width_ = 1;
        size_t height_ = 1;
        size_t depth_ = 1;
        bool initialized = false;
        DevicePtr device_ = nullptr;
        DataPtr data = std::make_shared<void *>(nullptr);
        const Backend &backend = cle::BackendManager::getInstance().getBackend();
    };

} // namespace cle

#endif // __INCLUDE_ARRAY_HPP
