#include "array.hpp"
#include "backend.hpp"
#include "device.hpp"
#include "tier1.hpp"
#include "utils.hpp"

#include <assert.h>

template <class T>
auto
run_gaussian_blur(const cle::mType & type) -> bool
{
  std::cout << cle::BackendManager::getInstance().getBackend().getType() << " test (" << type << ")" << std::endl;
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("TX", "all");

  static const size_t w = 4;
  static const size_t h = 4;
  static const size_t d = 1;
  std::vector<T>      input(w * h * d, static_cast<T>(0));
  std::vector<T>      valid(input.size(), static_cast<T>(0));
  std::vector<T>      output(input.size(), static_cast<T>(0));
  const size_t        center = (w / 2) + (h / 2) * w + (d / 2) * h * w;
  input[center] = 100;

  valid = { static_cast<T>(0.291504F), static_cast<T>(1.30643F), static_cast<T>(2.15394F), static_cast<T>(1.30643F),
            static_cast<T>(1.30643F),  static_cast<T>(5.85502F), static_cast<T>(9.65329F), static_cast<T>(5.85502F),
            static_cast<T>(2.15394F),  static_cast<T>(9.65329F), static_cast<T>(15.9156F), static_cast<T>(9.65329F),
            static_cast<T>(1.30643F),  static_cast<T>(5.85502F), static_cast<T>(9.65329F), static_cast<T>(5.85502F) };

  cle::Array gpu_input(w, h, d, cle::toType<T>(), type, input.data(), device);
  cle::Array gpu_output(w, h, d, cle::toType<T>(), type, device);

  cle::tier1::gaussian_blur_func(gpu_input, gpu_output, 1, 1, 1, device);

  gpu_output.read(output.data());
  for (auto && i : output)
  {
    std::cout << i << " ";
  }
  std::cout << std::endl;
  for (auto && i : valid)
  {
    std::cout << i << " ";
  }
  std::cout << std::endl;


  return std::equal(output.begin(), output.end(), valid.begin()) ? 0 : 1;
}

auto
main(int argc, char const * argv[]) -> int
{
  using T = float;

  cle::BackendManager::getInstance().setBackend(false);
  run_gaussian_blur<T>(cle::mType::Buffer);
  // assert(run_gaussian_blur<T>(cle::mType::Buffer) == 0);
  // assert(run_gaussian_blur<T>(cle::mType::Image) == 0);

  cle::BackendManager::getInstance().setBackend(true);
  run_gaussian_blur<T>(cle::mType::Buffer);
  // assert(run_gaussian_blur<T>(cle::mType::Buffer) == 0);
  // assert(run_gaussian_blur<T>(cle::mType::Image) == 0);
}
