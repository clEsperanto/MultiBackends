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

  std::array<size_t, 3> shape = { 3, 3, 3 };
  std::vector<float>    input(shape[0] * shape[1] * shape[2]);
  std::fill(input.begin(), input.end(), static_cast<float>(0));
  const int center = (shape[0] / 2) + (shape[1] / 2) * shape[0] + (shape[2] / 2) * shape[1] * shape[0];
  input[center] = 100;

  std::cout << "Input: ";
  for (size_t i = 0; i < input.size(); i++)
  {
    std::cout << input[i] << " ";
  }
  std::cout << std::endl;

  cle::Array gpu_input(shape[0], shape[1], shape[2], cle::dType::Float, cle::mType::Buffer, input.data(), device);
  cle::Array gpu_output(shape[0], shape[1], shape[2], cle::dType::Float, cle::mType::Buffer, device);

  cle::tier1::gaussian_blur_func(gpu_input, gpu_output, 1, 1, 1, device);

  std::vector<float> output_data(input.size());
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
}
