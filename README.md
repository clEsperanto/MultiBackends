# GPUbackends

Playground repo for CUDA and OpenCL Backend development.

## Roadmap

- [x] CMake build system
- Device (CUDA, OpenCL)
    - [x] Device initialisation
    - [x] Device info
- Backends (CUDA, OpenCL)
    - [x] Device search
    - [x] Device selection
    - [x] Buffer (create, write, read, free, copy)
    - [ ] Image (create, write, read, free, copy)
    - [ ] Program (ocl) / Module (cuda) creation from source
    - [ ] Kernel launch
- [x] Backend Manager
- Array (CUDA, OpenCL)
    - [x] Array creation
    - [x] Array allocate to device (create)
    - [x] Array copy to host (read)
    - [x] Array copy to device (write)
    - [ ] Array copy device to device (copy)
    - [x] Array free
- Image (CUDA, OpenCL)
    - [ ] Image creation
    - [ ] Image allocate to device (create)
    - [ ] Image copy to host (read)
    - [ ] Image copy to device (write)
    - [ ] Image copy device to device (copy)
    - [ ] Image free
- [ ] Defines Generator
- [ ] Preamble Generator
- [ ] Argument manager

# Install

Git clone the repository and run cmake as followed:

```bach
cmake -S . -B ./build
cmake --build ./build --parallel 10
```

## Requirements

- CMake
- C++ compiler
- CUDA and/or OpenCL installed on the system

