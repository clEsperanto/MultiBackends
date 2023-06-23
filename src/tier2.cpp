#include "tier2.hpp"
#include "tier1.hpp"

namespace cle::tier2
{

auto
difference_of_gaussian_func(const Device::Pointer &      device,
                            const Array &                src,
                            const std::optional<Array> & dst,
                            const float &                sigma1_x,
                            const float &                sigma1_y,
                            const float &                sigma1_z,
                            const float &                sigma2_x,
                            const float &                sigma2_y,
                            const float &                sigma2_z) -> const Array
{
  Array result;
  Array gauss1(src);
  Array gauss2(src);

  tier1::gaussian_blur_func(device, src, gauss1, sigma1_x, sigma1_y, sigma1_z);
  tier1::gaussian_blur_func(device, src, gauss2, sigma2_x, sigma2_y, sigma2_z);

  result = dst.value_or(Array(src));

  tier1::add_images_weighted_func(device, gauss1, gauss2, result, 1, -1);

  std::cout << "result count: " << result.get_count() << std::endl;
  return result;
}

} // namespace cle::tier2