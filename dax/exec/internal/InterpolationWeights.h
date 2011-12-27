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

  weights[0] = rcoords[0] * rcoords[1] * rcoords[2];
  weights[1] = pcoords[0] * rcoords[1] * rcoords[2];
  weights[2] = pcoords[0] * pcoords[1] * rcoords[2];
  weights[3] = rcoords[0] * pcoords[1] * rcoords[2];
  weights[4] = rcoords[0] * rcoords[1] * pcoords[2];
  weights[5] = pcoords[0] * rcoords[1] * pcoords[2];
  weights[6] = pcoords[0] * pcoords[1] * pcoords[2];
  weights[7] = rcoords[0] * pcoords[1] * pcoords[2];
}

}}}

#endif //__dax__exec__internal__InterpolationWeights_h
