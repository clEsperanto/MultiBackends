#include "array.hpp"
#include "backend.hpp"
#include "device.hpp"
#include "execution.hpp"
#include "utils.hpp"

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

  static const size_t w = 5;
  static const size_t h = 5;
  static const size_t d = 2;

  std::cout << "Allocate memory" << std::endl;
  cle::Array gpu_arr1(w, h, d, cle::dType::Int32, cle::mType::Buffer, device);

  std::cout << "Write memory" << std::endl;
  std::vector<int> data(w * h * d);
  for (int i = 0; i < data.size(); i++)
  {
    data[i] = i;
  }
  cle::Array gpu_arr2(w, h, d, cle::dType::Int32, cle::mType::Buffer, data.data(), device);

  std::cout << "Read memory" << std::endl;
  std::vector<int> data_out(w * h * d);
  for (int i = 0; i < data_out.size(); i++)
  {
    data_out[i] = -i;
    std::cout << data_out[i] << " ";
  }
  std::cout << std::endl;
  gpu_arr2.read(data_out.data());
  for (int i = 0; i < data_out.size(); i++)
  {
    std::cout << data_out[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Write existing memory" << std::endl;
  gpu_arr1.write(data.data());
  std::vector<int> data_out2(w * h * d);
  gpu_arr1.read(data_out2.data());
  for (int i = 0; i < data_out2.size(); i++)
  {
    std::cout << data_out2[i] << " ";
  }
  std::cout << std::endl;

  std::cout << "Copy memory" << std::endl;
  cle::Array       gpu_arr3(5, 5, 2, cle::dType::Int32, cle::mType::Buffer, device);
  std::vector<int> data_out3(w * h * d);
  gpu_arr1.copy(gpu_arr3);
  gpu_arr3.read(data_out3.data());
  for (int i = 0; i < data_out3.size(); i++)
  {
    std::cout << data_out3[i] << " ";
  }
  std::cout << std::endl;
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