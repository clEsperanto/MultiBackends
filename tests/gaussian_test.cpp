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

  std::array<size_t, 3> shape = { 4, 4, 1 };
  std::vector<float>    input(shape[0] * shape[1] * shape[2]);
  std::vector<float>    valid_data(shape[0] * shape[1] * shape[2]);
  std::vector<float>    output_data(input.size());
  std::fill(input.begin(), input.end(), static_cast<float>(0));
  std::fill(valid_data.begin(), valid_data.end(), static_cast<float>(0));
  const int center = (shape[0] / 2) + (shape[1] / 2) * shape[0] + (shape[2] / 2) * shape[1] * shape[0];
  input[center] = 100;

  valid_data = { static_cast<float>(0.291504F), static_cast<float>(1.30643F), static_cast<float>(2.15394F),
            static_cast<float>(1.30643F),  static_cast<float>(1.30643F), static_cast<float>(5.85502F),
            static_cast<float>(9.65329F),  static_cast<float>(5.85502F), static_cast<float>(2.15394F),
            static_cast<float>(9.65329F),  static_cast<float>(15.9156F), static_cast<float>(9.65329F),
            static_cast<float>(1.30643F),  static_cast<float>(5.85502F), static_cast<float>(9.65329F),
            static_cast<float>(5.85502F) };

  cle::Array gpu_input(shape[0], shape[1], shape[2], cle::dType::Float, cle::mType::Buffer, input.data(), device);
  cle::Array gpu_output(shape[0], shape[1], shape[2], cle::dType::Float, cle::mType::Buffer, device);

  cle::tier1::gaussian_blur_func(gpu_input, gpu_output, 1, 1, 1, device);

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
