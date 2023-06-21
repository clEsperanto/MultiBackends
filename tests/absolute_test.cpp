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

  std::vector<float> valid_data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
  std::vector<float> input_data = { -1, -2, -3, -4, -5, -6, -7, -8, -9, -1, -2, -3, -4, -5, -6, -7, -8, -9 };
  cle::Array         gpu_input(3, 3, 2, cle::dType::Float, cle::mType::Buffer, input_data.data(), device);
  cle::Array         gpu_output(3, 3, 2, cle::dType::Float, cle::mType::Buffer, device);
  gpu_input.copy(gpu_output);

  cle::tier1::absolute_func(gpu_input, gpu_output, device);

  std::vector<float> output_data(input_data.size());
  gpu_output.read(output_data.data());

  return std::equal(output_data.begin(), output_data.end(), valid_data.begin());
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
