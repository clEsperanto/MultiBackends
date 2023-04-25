#include "backend.hpp"
#include "device.hpp"

void ocl_list_devices()
{
    std::cout << "OpenCL backend" << std::endl;
    cle::BackendManager &backendManager = cle::BackendManager::getInstance();
    backendManager.selectBackend(false); // Use OpenCL

    auto device_list = backendManager.getBackend().getDevicesList();
    std::cout << "Device list:" << std::endl;
    for (auto &&i : device_list)
    {
        std::cout << "\t" << i << std::endl;
    }

    auto device = backendManager.getBackend().getDevice("TX");
    std::cout << "Selected Device :" << device->getName() << std::endl;
    std::cout << "Device Info :" << device->getInfo() << std::endl;
    device->initialize();

    std::cout << "Allocate memory" << std::endl;
    size_t size = 128 * sizeof(float);
    cl::Memory *data_ptr;
    backendManager.getBackend().allocateMemory(device, size, (void **)&data_ptr);
    std::cout << data_ptr->getInfo<CL_MEM_SIZE>() << " == " << 128 * sizeof(float) << std::endl;
    std::cout << data_ptr->getInfo<CL_MEM_TYPE>() << " == " << CL_MEM_OBJECT_BUFFER << std::endl;

    std::cout << "Write memory" << std::endl;
    float data[128];
    for (int i = 0; i < 128; i++)
    {
        data[i] = i;
    }
    backendManager.getBackend().writeMemory(device, (void **)&data_ptr, size, data);

    std::cout << "Read memory" << std::endl;
    float data_out[128];
    for (int i = 0; i < 128; i++)
    {
        data_out[i] = -i;
        std::cout << data_out[i] << " ";
    }
    std::cout << std::endl;
    backendManager.getBackend().readMemory(device, (const void **)&data_ptr, size, data_out);
    for (int i = 0; i < 128; i++)
    {
        std::cout << data_out[i] << " ";
    }
    std::cout << std::endl;

    // std::cout << "Free memory" << std::endl;
    // backendManager.getBackend().freeMemory(device, (void **)&data_ptr);
}

void cuda_list_devices()
{
    std::cout << "CUDA backend" << std::endl;
    cle::BackendManager &backendManager = cle::BackendManager::getInstance();
    backendManager.selectBackend(true); // Use CUDA

    std::vector<std::string> device_list = backendManager.getBackend().getDevicesList();
    std::cout << "Device list:" << std::endl;
    for (auto &&i : device_list)
    {
        std::cout << "\t" << i << std::endl;
    }

    auto device = backendManager.getBackend().getDevice("TX");
    std::cout << "Selected Device :" << device->getName() << std::endl;
    std::cout << "Device Info :" << device->getInfo() << std::endl;
    device->initialize();

    std::cout << "Allocate memory" << std::endl;
    size_t size = 128 * sizeof(float);
    float *data_ptr;
    backendManager.getBackend().allocateMemory(device, size, (void **)&data_ptr);

    std::cout << "Write memory" << std::endl;
    float data[128];
    for (int i = 0; i < 128; i++)
    {
        data[i] = i;
    }
    backendManager.getBackend().writeMemory(device, (void **)&data_ptr, size, data);

    std::cout << "Read memory" << std::endl;
    float data_out[128];
    for (int i = 0; i < 128; i++)
    {
        data_out[i] = -i;
        std::cout << data_out[i] << " ";
    }
    std::cout << std::endl;
    backendManager.getBackend().readMemory(device, (const void **)&data_ptr, size, data_out);
    for (int i = 0; i < 128; i++)
    {
        std::cout << data_out[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Free memory" << std::endl;
    backendManager.getBackend().freeMemory(device, (void **)&data_ptr);
}

int main(int argc, char **argv)
{
    ocl_list_devices();
    cuda_list_devices();
}