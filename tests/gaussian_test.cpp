#include "array.hpp"
#include "backend.hpp"
#include "device.hpp"
#include "tier1.hpp"
#include "utils.hpp"

#include <assert.h>
#include <iomanip>
#include <limits>

template <class T>
auto
run_gaussian_blur(const cle::mType & type) -> bool
{
  std::cout << cle::BackendManager::getInstance().getBackend().getType() << " test (" << type << ")" << std::endl;
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("TX", "all");

  static const size_t w = 5;
  static const size_t h = 5;
  static const size_t d = 1;
  std::vector<T>      input(w * h * d, static_cast<T>(0));
  std::vector<T>      valid(input.size(), static_cast<T>(0));
  std::vector<T>      output(input.size(), static_cast<T>(0));
  const size_t        center = (w / 2) + (h / 2) * w + (d / 2) * h * w;
  input[center] = 100;

  valid = { static_cast<T>(0.2915041744709014892578125F), static_cast<T>(1.30643117427825927734375F),
            static_cast<T>(2.1539404392242431640625F),    static_cast<T>(1.30643117427825927734375F),
            static_cast<T>(0.2915041744709014892578125F), static_cast<T>(1.3064310550689697265625F),
            static_cast<T>(5.855018138885498046875F),     static_cast<T>(9.6532917022705078125F),
            static_cast<T>(5.855018138885498046875F),     static_cast<T>(1.3064310550689697265625F),
            static_cast<T>(2.153940677642822265625F),     static_cast<T>(9.65329265594482421875F),
            static_cast<T>(15.91558742523193359375F),     static_cast<T>(9.65329265594482421875F),
            static_cast<T>(2.153940677642822265625F),     static_cast<T>(1.3064310550689697265625F),
            static_cast<T>(5.855018138885498046875F),     static_cast<T>(9.6532917022705078125F),
            static_cast<T>(5.855018138885498046875F),     static_cast<T>(1.3064310550689697265625F),
            static_cast<T>(0.2915041744709014892578125F), static_cast<T>(1.30643117427825927734375F),
            static_cast<T>(2.1539404392242431640625F),    static_cast<T>(1.30643117427825927734375F),
            static_cast<T>(0.2915041744709014892578125F) };


  auto gpu_input = cle::Array::create(w, h, d, cle::toType<T>(), type, input.data(), device);
  auto gpu_output = cle::Array::create(w, h, d, cle::toType<T>(), type, device);

  cle::tier1::gaussian_blur_func(device, gpu_input, gpu_output, 1, 1, 1);

  gpu_output->read(output.data());

  return std::equal(output.begin(), output.end(), valid.begin()) ? 0 : 1;
}

auto
main(int argc, char const * argv[]) -> int
{
  using T = float;

  cle::BackendManager::getInstance().setBackend(false);
  assert(run_gaussian_blur<T>(cle::mType::Buffer) == 0);
  assert(run_gaussian_blur<T>(cle::mType::Image) == 0);

  cle::BackendManager::getInstance().setBackend(true);
  assert(run_gaussian_blur<T>(cle::mType::Buffer) == 0);
}
