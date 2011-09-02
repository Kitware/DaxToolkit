/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#ifndef __dax__exec__internal__DerviativeWeights_h
#define __dax__exec__internal__DerviativeWeights_h

#include <dax/Types.h>

namespace dax {
namespace exec {
namespace internal {

DAX_EXEC_EXPORT static void derivativeWeightsVoxel(const dax::Vector3 &pcoords,
                                                   dax::Vector3 weights[8])
{
  dax::Vector3 rcoords = dax::make_Vector3(1, 1, 1) - pcoords;

  weights[0].x = -rcoords.y*rcoords.z;
  weights[0].y = -rcoords.x*rcoords.z;
  weights[0].z = -rcoords.x*rcoords.y;

  weights[1].x = rcoords.y*rcoords.z;
  weights[1].y = -pcoords.x*rcoords.z;
  weights[1].z = -pcoords.x*rcoords.y;

  weights[2].x = pcoords.y*rcoords.z;
  weights[2].y = pcoords.x*rcoords.z;
  weights[2].z = -pcoords.x*pcoords.y;

  weights[3].x = -pcoords.y*rcoords.z;
  weights[3].y = rcoords.x*rcoords.z;
  weights[3].z = -rcoords.x*pcoords.y;

  weights[4].x = -rcoords.y*pcoords.z;
  weights[4].y = -rcoords.x*pcoords.z;
  weights[4].z = rcoords.x*rcoords.y;

  weights[5].x = rcoords.y*pcoords.z;
  weights[5].y = -pcoords.x*pcoords.z;
  weights[5].z = pcoords.x*rcoords.y;

  weights[6].x = pcoords.y*pcoords.z;
  weights[6].y = pcoords.x*pcoords.z;
  weights[6].z = pcoords.x*pcoords.y;

  weights[7].x = -pcoords.y*pcoords.z;
  weights[7].y = rcoords.x*pcoords.z;
  weights[7].z = rcoords.x*pcoords.y;
}

}}}

#endif //__dax__exec__internal__DerviativeWeights_h
