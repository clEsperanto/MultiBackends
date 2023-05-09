# Multi-Backends

Playground repo for CUDA and OpenCL Backend development.

## Roadmap

- [x] CMake build system
- Device (CUDA, OpenCL)
    - [x] Device initialisation
    - [x] Device info
    - [x] Device program cache
- Backends (CUDA, OpenCL)
    - [x] Device search
    - [x] Device selection
    - [x] Buffer 
        - [x] Buffer allocate to device (create)
        - [x] Buffer copy to host (read)
        - [x] Buffer copy to device (write)
        - [x] Buffer copy device to device (copy)
        - [x] Buffer free
    - [ ] Image (1D/2D/3D)
        - [ ] Image allocate to device (create)
        - [ ] Image copy to host (read)
        - [ ] Image copy to device (write)
        - [ ] Image copy device to device (copy)
        - [ ] Image free
    - [x] Program (ocl) / Module (cuda) creation from source
    - [ ] Kernel launch
- [x] Backend Manager (singleton)
- Array (CUDA, OpenCL)
    - [x] Array creation
    - [x] Array allocate to device (create)
    - [x] Array copy to host (read)
    - [x] Array copy to device (write)
    - [x] Array copy device to device (copy)
    - [x] Array free
- Image (CUDA, OpenCL)
    - [ ] Image creation
    - [ ] Image allocate to device (create)
    - [ ] Image copy to host (read)
    - [ ] Image copy to device (write)
    - [ ] Image copy device to device (copy)
    - [ ] Image free
- Execution
    - [ ] Generate Defines
    - [ ] Get Preamble
    - [ ] Manage kernel parameters
    - [ ] Launch kernel

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

