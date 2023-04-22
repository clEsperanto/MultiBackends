#ifndef __INCLUDE_IMAGE_HPP
#define __INCLUDE_IMAGE_HPP

#include <memory>
#include <string>
#include <stdexcept>

#include <CL/cl.hpp>
#include <cuda_runtime_api.h>

#include "device.hpp"
#include "backend.hpp"

namespace cle
{

class Image {
public:
    Image() : width_(0), height_(0), depth_(0), bytes_per_pixel_(0), size_(0), is_opencl_(false), is_cuda_(false) {}
    Image(size_t width, size_t height, size_t depth, size_t bytes_per_pixel) : width_(width), height_(height), depth_(depth) bytes_per_pixel_(bytes_per_pixel), is_opencl_(false), is_cuda_(false) {}

    ~Image() {
        if (is_cuda_) {
            BackendManager::getBackend()->freeMemory(device_.get(), std::get<void*>(memory_));
        } else if (is_opencl_) {
            BackendManager::getBackend()->freeMemory(device_.get(), std::get<cl::Memory>(memory_).get());
        }        
    }

    void createCUDA(const cle::Device& device) {
        if (is_opencl_) {
            throw std::runtime_error("Image is already allocated on OpenCL, cannot allocate on CUDA");
        }
        if (auto memory_ptr = std::get_if<void*>(&memory_)) {
            BackendManager::getBackend()->allocateMemory(device, width_*height_*depth_*bytes_per_pixels_, *memory_ptr)
            device_ = std::make_shared<CUDADevice>(device);
        }
    }

    void createOpenCL(const cle::Device& device) {
        if (is_cuda_) {
            throw std::runtime_error("Image is already allocated on OpenCL, cannot allocate on CUDA");
        }
        if (auto memory_ptr = std::get_if<cl::Memory>(&memory_).get()) {
            BackendManager::getBackend()->allocateMemory(device, width_*height_*depth_*bytes_per_pixels_, memory_ptr->get())
            device_ = std::make_shared<OpenCLDevice>(device);
        }
    }

    size_t width() const { return width_; }
    size_t height() const { return height_; }
    size_t depth() const { return depth_; }
    size_t bytesPerPixel() const { return bytes_per_pixel_; }

private:

    size_t width_;
    size_t height_;
    size_t depth_;
    size_t bytes_per_pixel_;

    bool is_cuda_;
    bool is_opencl_;
    
    std::variant<cl::Memory, void*> memory_;
    std::shared_ptr<Device> device_;
};



} // namespace cle

#endif //__INCLUDE_IMAGE_HPP