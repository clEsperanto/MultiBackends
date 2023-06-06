#include "memory.hpp"

void
run_test()
{
  auto device = cle::BackendManager::getInstance().getBackend().getDevice("TX", "all");
  device->initialize();
  std::cout << "run_test : " << device->getName() << std::endl;

  std::cout << "make host data" << std::endl;
  static const size_t size = 5 * 5 * 2;
  float *             data = new float[size];
  for (int i = 0; i < size; i++)
  {
    data[i] = i;
  }
  auto arr_empty = cle::memory::create<float>(5, 5, 2, device);

  std::cout << "push it" << std::endl;
  auto arr = cle::memory::push(data, 5, 5, 2, device);

  float * data_out = new float[arr.nbElements()];
  std::fill(data_out, data_out + size, -1);

  auto arr_copy = arr;

  std::cout << "arr: " << arr << std::endl;
  std::cout << "arr_copy: " << arr_copy << std::endl;

  cle::memory::pull(arr_copy, data_out);

  for (int i = 0; i < size; i++)
  {
    std::cout << data_out[i] << " ";
  }
  std::cout << std::endl;
}

int
main(int argc, char ** argv)
{
  cle::BackendManager::getInstance().setBackend(false);
  std::cout << cle::BackendManager::getInstance().getBackend().getType() << std::endl;
  run_test();

  cle::BackendManager::getInstance().setBackend(true);
  std::cout << cle::BackendManager::getInstance().getBackend().getType() << std::endl;
  run_test();

  return 0;
}