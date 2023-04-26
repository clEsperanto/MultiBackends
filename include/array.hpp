#ifndef __INCLUDE_ARRAY_HPP
#define __INCLUDE_ARRAY_HPP

#include "backend.hpp"
#include "device.hpp"

#include <variant>

namespace cle
{

    class Array
    {
    public:
        using DevicePtr = std::shared_ptr<cle::Device>;
        using DataPtr = std::shared_ptr<void *>;

        Array() = default;
        Array(size_t width, size_t height, size_t depth, size_t bytes_per_element) : width_(width), height_(height), depth_(depth), bytes_per_element_(bytes_per_element)
        {
            this->device_ = nullptr;
            this->data = std::make_shared<void *>(nullptr);
        }

        Array(size_t width, size_t height, size_t depth, size_t bytes_per_element, const DevicePtr &device) : Array(width, height, depth, bytes_per_element)
        {
            this->device_ = device;
            this->data = std::make_shared<void *>(nullptr);
            backend.allocateMemory(device_, width_ * height_ * depth_ * bytes_per_element_, this->get());
            this->initialized = true;
        }

        Array(size_t width, size_t height, size_t depth, size_t bytes_per_element, const void *host_data, const DevicePtr &device) : Array(width, height, depth, bytes_per_element, device)
        {
            backend.writeMemory(device_, this->get(), width_ * height_ * depth_ * bytes_per_element_, host_data);
        }

        ~Array()
        {
            if (this->initialized)
            {
                backend.freeMemory(device_, this->get());
            }
        }

        auto allocate() -> void
        {
            if (!this->initialized)
            {
                backend.allocateMemory(device_, width_ * height_ * depth_ * bytes_per_element_, this->get());
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
            backend.writeMemory(device_, this->get(), width_ * height_ * depth_ * bytes_per_element_, host_data);
        }

        auto read(void *host_data) const -> void
        {
            backend.readMemory(device_, this->c_get(), width_ * height_ * depth_ * bytes_per_element_, host_data);
        }

        [[nodiscard]] auto nbElements() const -> size_t { return width_ * height_ * depth_; }
        [[nodiscard]] auto width() const -> size_t { return width_; }
        [[nodiscard]] auto height() const -> size_t { return height_; }
        [[nodiscard]] auto depth() const -> size_t { return depth_; }
        [[nodiscard]] auto bytesPerElements() const -> size_t { return bytes_per_element_; }
        [[nodiscard]] auto device() const -> DevicePtr { return device_; }

        friend auto operator<<(std::ostream &out, const Array &array) -> std::ostream &
        {
            out << "Array ([" << array.width_ << "," << array.height_ << "," << array.depth_ << "], dtype=" << array.bytes_per_element_ << ")";
            return out;
        }

    protected:
        [[nodiscard]] auto get() const -> void ** { return data.get(); }
        [[nodiscard]] auto c_get() const -> const void ** { return (const void **)data.get(); }

    private:
        bool initialized = false;
        size_t width_ = 1;
        size_t height_ = 1;
        size_t depth_ = 1;
        size_t bytes_per_element_ = sizeof(float);
        DevicePtr device_;
        DataPtr data;
        const Backend &backend = cle::BackendManager::getInstance().getBackend();
    };

} // namespace cle

#endif // __INCLUDE_ARRAY_HPP
