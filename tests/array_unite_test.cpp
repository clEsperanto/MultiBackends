#include "array.hpp"
#include "backend.hpp"
#include "utils.hpp"

template <class T>
int
data_test(cle::mType type1, cle::mType type2)
{
  std::cout << cle::BackendManager::getInstance().getBackend().getType() << " test (" << type1 << "," << type2 << ")"
            << std::endl;
  cle::BackendManager & backendManager = cle::BackendManager::getInstance();
  auto                  device = backendManager.getBackend().getDevice("TX", "all");
  device->initialize();

  static const size_t w = 5;
  static const size_t h = 4;
  static const size_t d = 3;
  std::vector<T>      input(w * h * d, -5);
  std::vector<T>      output(w * h * d, -10);

  std::cout << "\tcreating bufferA" << std::endl;
  cle::Array bufferA(w, h, d, cle::toType<T>(), type1, device);
  std::cout << "\twrite into bufferA" << std::endl;
  bufferA.write(input.data());
  std::cout << "\tfill bufferA" << std::endl;
  bufferA.fill(15);
  std::cout << "\tcreating bufferB and write into it" << std::endl;
  cle::Array bufferB(w, h, d, cle::toType<T>(), type2, output.data(), device);
  std::cout << "\tcopy bufferA into bufferB" << std::endl;
  bufferA.copy(bufferB);
  std::cout << "\tread bufferB" << std::endl;
  bufferB.read(output.data());
  std::cout << bufferB << std::endl;
  for (auto & i : output)
  {
    std::cout << i << " ";
  }
  std::cout << std::endl;

  std::equal(input.begin(), input.end(), output.begin()) ? std::cout << "success" << std::endl
                                                         : std::cout << "fail" << std::endl;
  return 0;
}

int
main(int argc, char ** argv)
{
  using T = float;

  cle::BackendManager::getInstance().setBackend(false);
  data_test<T>(cle::mType::Buffer, cle::mType::Buffer);
  // data_test<T>(cle::mType::Image, cle::mType::Image);
  // data_test<T>(cle::mType::Buffer, cle::mType::Image);
  // data_test<T>(cle::mType::Image, cle::mType::Buffer);

  // cle::BackendManager::getInstance().setBackend(true);
  // data_test<T>(cle::mType::Buffer, cle::mType::Buffer);
  // data_test<T>(cle::mType::Image, cle::mType::Image);
  // data_test<T>(cle::mType::Buffer, cle::mType::Image);
  // data_test<T>(cle::mType::Image, cle::mType::Buffer);

  return 0;
}
