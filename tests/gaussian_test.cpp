#include "array.hpp"
#include "backend.hpp"
#include "device.hpp"
#include "tier1.hpp"
#include "utils.hpp"

auto
run_absolute_test() -> bool
{
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("TX", "all");
  device->initialize();

  std::vector<float> input = { 0, 0, 0, 0, 1, 0, 0, 0, 0 };
  std::vector<float> output_data(input.size());

  cle::Array gpu_input(9, 1, 1, cle::dType::Float, cle::mType::Buffer, input.data(), device);
  cle::Array gpu_output(9, 1, 1, cle::dType::Float, cle::mType::Buffer, device);

  cle::tier1::gaussian_blur_func(gpu_input, gpu_output, 1, 0, 0, device);

  gpu_output.read(output_data.data());

  std::cout << "Output: ";
  for (size_t i = 0; i < output_data.size(); i++)
  {
    std::cout << output_data[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}

auto
main(int argc, char const * argv[]) -> int
{
  cle::BackendManager::getInstance().setBackend(false);
  if (!run_absolute_test())
  {
    return 0;
  }

  cle::BackendManager::getInstance().setBackend(true);
  if (!run_absolute_test())
  {
    return 0;
  }
}
