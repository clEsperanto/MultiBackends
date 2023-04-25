#ifndef __INCLUDE_MEMORY_HPP
#define __INCLUDE_MEMORY_HPP

#include "array.hpp"
#include "backend.hpp"
#include "device.hpp"

namespace cle
{
    namespace memory
    {
        using DevicePtr = std::shared_ptr<cle::Device>;

        template <typename T>
        auto create(size_t width, size_t height, size_t depth, const DevicePtr &device) -> Array
        {
            return Array{width, height, depth, sizeof(T), device};
        }

        template <typename T>
        auto push(const T *host_data, size_t width, size_t height, size_t depth, const DevicePtr &device) -> Array
        {
            // auto arr = create<T>(width, height, depth, device);
            // arr.write(host_data);
            // return arr;
            return Array{width, height, depth, sizeof(T), host_data, device};
        }

        template <typename T>
        auto pull(const Array &arr, T *host_arr) -> void
        {
            // size = arr.width() * arr.height() * arr.depth();
            // T *host_arr = new T[arr.width() * arr.height() * arr.depth()];
            arr.read(host_arr);
            // host_arr = host_data;
        }

    } // namespace memory
} // namespace cle

#endif // __INCLUDE_MEMORY_HPP
