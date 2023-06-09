#include "array.hpp"
#include "backend.hpp"
#include "device.hpp"
#include "execution.hpp"
#include "memory.hpp"
#include "utils.hpp"
#include <variant>


void
run_test()
{

  cle::BackendManager & backendManager = cle::BackendManager::getInstance();
  auto                  device_list = backendManager.getBackend().getDevicesList("all");
  std::cout << "Device list:" << std::endl;
  for (auto && i : device_list)
  {
    std::cout << "\t" << i << std::endl;
  }

  auto device = backendManager.getBackend().getDevice("TX", "all");
  std::cout << "Selected Device :" << device->getName() << std::endl;
  std::cout << "Device Info :" << device->getInfo() << std::endl;
  device->initialize();

  static const size_t size = 5 * 5 * 2;
  float *             data = new float[size];
  for (int i = 0; i < size; i++)
  {
    data[i] = i;
  }

  cle::Array        gpu_arr(5, 5, 2, cle::dType::Float, cle::mType::Image, data, device);
  cle::ParameterMap parameters;
  cle::ConstantMap  constants{ { "CLK_NORMALIZED_COORDS_FALSE", 1 },
                               { "CLK_ADDRESS_CLAMP_TO_EDGE", 2 },
                               { "CLK_FILTER_NEAREST", 4 } };
  cle::KernelInfo   kernel;
  cle::RangeArray   global_rage;

  parameters.emplace("src", gpu_arr);
  parameters.emplace("dst", gpu_arr);

  cle::execute(device, kernel, parameters, constants, global_rage);
}

int
main(int argc, char ** argv)
{
  cle::BackendManager::getInstance().setBackend(false);
  std::cout << cle::BackendManager::getInstance().getBackend().getType() << std::endl;
  run_test();

  cle::BackendManager::getInstance().setBackend(true);
  std::cout << cle::BackendManager::getInstance().getBackend().getType() << std::endl;
  run_test();

  return 0;
}