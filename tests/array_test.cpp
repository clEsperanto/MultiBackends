#include "array.hpp"
#include "backend.hpp"
#include "utils.hpp"

#include <assert.h>

template <class T>
int
run_array(cle::mType type1, cle::mType type2)
{
  std::cout << cle::BackendManager::getInstance().getBackend().getType() << " test (" << type1 << "," << type2 << ")"
            << std::endl;
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("TX", "all");

  static const size_t w = 5;
  static const size_t h = 4;
  static const size_t d = 3;
  std::vector<T>      input(w * h * d, -5);
  std::vector<T>      valid(input.size(), 12);
  std::vector<T>      output(input.size(), -10);

  cle::Array bufferA(w, h, d, cle::toType<T>(), type1, device);
  cle::Array bufferB(w, h, d, cle::toType<T>(), type2, output.data(), device);
  bufferA.write(input.data());
  bufferA.fill(12);
  bufferA.copy(bufferB);
  cle::Array bufferC(bufferB);

  cle::Array bufferD = bufferC;

  std::optional<cle::Array> bufferE(bufferD);

  bufferE.value().read(output.data());

  return std::equal(valid.begin(), valid.end(), output.begin()) ? 0 : 1;
}

int
main(int argc, char ** argv)
{
  using T = float;
  cle::BackendManager::getInstance().setBackend(false);
  assert(run_array<T>(cle::mType::Buffer, cle::mType::Buffer) == 0);
  assert(run_array<T>(cle::mType::Image, cle::mType::Image) == 0);
  assert(run_array<T>(cle::mType::Buffer, cle::mType::Image) == 0);
  assert(run_array<T>(cle::mType::Image, cle::mType::Buffer) == 0);

  cle::BackendManager::getInstance().setBackend(true);
  assert(run_array<T>(cle::mType::Buffer, cle::mType::Buffer) == 0);
  assert(run_array<T>(cle::mType::Image, cle::mType::Image) == 0);
  assert(run_array<T>(cle::mType::Buffer, cle::mType::Image) == 0);
  assert(run_array<T>(cle::mType::Image, cle::mType::Buffer) == 0);

  return 0;
}
