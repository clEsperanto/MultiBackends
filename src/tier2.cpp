#include "tier2.hpp"

namespace cle::tier2
{

auto
add_images_weighted_func(const Device::Pointer & device,
                         const Array &           src,
                         const Array &           dst,
                         const float &           sigma1_x,
                         const float &           sigma1_y,
                         const float &           sigma1_z,
                         const float &           sigma2_x,
                         const float &           sigma2_y,
                         const float &           sigma2_z) -> void
{
  Array gauss1(dst);
  Array gauss2(dst);

  tier1::gaussian_blur_func(device, src, gauss1, sigma1_x, sigma1_y, sigma1_z);
  tier1::gaussian_blur_func(device, src, gauss2, sigma2_x, sigma2_y, sigma2_z);
  tier1::add_images_weighted_func(device, gauss1, gauss2, dst, 1, -1);
}

} // namespace cle::tier2