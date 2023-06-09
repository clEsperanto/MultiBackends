#include "backend.hpp"

namespace cle
{

auto
BackendManager::getInstance() -> BackendManager &
{
  static BackendManager instance;
  return instance;
}

auto
BackendManager::setBackend(bool useCUDA) -> void
{
  if (useCUDA)
  {
    this->backend = std::make_unique<CUDABackend>();
  }
  else
  {
    this->backend = std::make_unique<OpenCLBackend>();
  }
}

auto
BackendManager::getBackend() const -> const Backend &
{
  if (!this->backend)
  {
    throw std::runtime_error("Backend not selected.");
  }
  return *this->backend;
}

} // namespace cle
