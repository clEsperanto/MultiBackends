#ifndef __INCLUDE_BACKENDMANAGER_HPP
#define __INCLUDE_BACKENDMANAGER_HPP

#include "backend.hpp"

namespace cle
{
class BackendManager
{
public:
  static auto
  getInstance() -> BackendManager &;

  auto
  setBackend(bool useCUDA) -> void;

  [[nodiscard]] auto
  getBackend() const -> const Backend &;

  friend auto
  operator<<(std::ostream & out, const BackendManager & backend_manager) -> std::ostream &
  {
    out << backend_manager.getBackend().getType() << " backend";
    return out;
  }

  auto
  operator=(const BackendManager &) -> BackendManager & = delete;
  BackendManager(const BackendManager &) = delete;

private:
  std::shared_ptr<Backend> backend;
  BackendManager() = default;
};

} // namespace cle

#endif // __INCLUDE_BACKENDMANAGER_HPP
