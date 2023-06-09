#include "array.hpp"
#include "backend.hpp"
#include "device.hpp"
#include "tier1.hpp"
#include "utils.hpp"

auto
run_absolute_test() -> void
{
  std::cout << "Run absolute test on backend : " << cle::BackendManager::getInstance().getBackend().getType()
            << std::endl;
  //   auto device = cle::BackendManager::getInstance().getBackend().getDevice("TX", "all");
  //   std::cout << "Selected Device :" << device->getName() << std::endl;
  //   std::cout << "Device Info :" << device->getInfo() << std::endl;
  //   device->initialize();

  //   std::vector<float> input_data = { -1, -2, -3, -4, -5, -6, -7, -8, -9 };
  //   cle::Array         gpu_input(3, 3, 1, cle::dType::Float, cle::mType::Buffer, input_data.data(), device);
  //   cle::Array         gpu_output(3, 3, 1, cle::dType::Float, cle::mType::Buffer, device);

  //   cle::tier1::absolute_func(gpu_input, gpu_output, device);

  //   std::vector<float> output_data(input_data.size());
  //   gpu_output.read(output_data.data());

  //   std::cout << "GPU output for absotule kernel: ";
  //   for (auto && i : output_data)
  //   {
  //     std::cout << i << " ";
  //   }
  //   std::cout << std::endl;
}

auto
main(int argc, char const * argv[]) -> int
{
  //   cle::BackendManager::getInstance().setBackend(false);
  //   run_absolute_test();
  return 0;
}
