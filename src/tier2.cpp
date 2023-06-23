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
  Array result = dst.value_or(Array(src));
  Array gauss1(result);
  Array gauss2(result);
  tier1::gaussian_blur_func(device, src, gauss1, sigma1_x, sigma1_y, sigma1_z);
  tier1::gaussian_blur_func(device, src, gauss2, sigma2_x, sigma2_y, sigma2_z);
  tier1::add_images_weighted_func(device, gauss1, gauss2, result, 1, -1);
  return result;
}

} // namespace cle::tier2