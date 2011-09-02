/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax__exec__internal__InterpolationWeights_h
#define __dax__exec__internal__InterpolationWeights_h

#include <dax/Types.h>

namespace dax {
namespace exec {
namespace internal {

DAX_EXEC_EXPORT static void interpolationWeightsVoxel(
  const dax::Vector3 &pcoords,
  dax::Scalar weights[8])
{
  dax::Vector3 rcoords = dax::make_Vector3(1, 1, 1) - pcoords;

  weights[0] = rcoords.x * rcoords.y * rcoords.z;
  weights[1] = pcoords.x * rcoords.y * rcoords.z;
  weights[2] = pcoords.x * pcoords.y * rcoords.z;
  weights[3] = rcoords.x * pcoords.y * rcoords.z;
  weights[4] = rcoords.x * rcoords.y * pcoords.z;
  weights[5] = pcoords.x * rcoords.y * pcoords.z;
  weights[6] = pcoords.x * pcoords.y * pcoords.z;
  weights[7] = rcoords.x * pcoords.y * pcoords.z;
}

}}}

#endif //__dax__exec__internal__InterpolationWeights_h
