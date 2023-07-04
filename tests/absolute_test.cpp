#include "array.hpp"
#include "backend.hpp"
#include "device.hpp"
#include "tier1.hpp"
#include "utils.hpp"

#include <assert.h>

template <class T>
auto
run_absolute(cle::mType type) -> bool
{
  std::cout << cle::BackendManager::getInstance().getBackend().getType() << " test (" << type << ")" << std::endl;
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("TX", "all");

  static const size_t w = 5;
  static const size_t h = 4;
  static const size_t d = 3;
  std::vector<T>      input(w * h * d, -5);
  std::vector<T>      output(w * h * d, -5);
  std::vector<T>      valid(w * h * d, 5);

  auto gpu_input = cle::Array::create(w, h, d, cle::toType<T>(), type, input.data(), device);
  auto gpu_output = cle::tier1::absolute_func(device, gpu_input, nullptr);
  gpu_output->read(output.data());

  return std::equal(output.begin(), output.end(), valid.begin()) ? 0 : 1;
}

auto
main(int argc, char const * argv[]) -> int
{
  using T = float;

  cle::BackendManager::getInstance().setBackend("opencl");
  assert(run_absolute<T>(cle::mType::BUFFER) == 0);
  assert(run_absolute<T>(cle::mType::IMAGE) == 0);

  cle::BackendManager::getInstance().setBackend("cuda");
  assert(run_absolute<T>(cle::mType::BUFFER) == 0);
  // assert(run_absolute<T>(cle::mType::IMAGE) == 0);
}
