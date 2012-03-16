//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2012 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __dax__exec__internal__DerviativeWeights_h
#define __dax__exec__internal__DerviativeWeights_h

#include <dax/Types.h>

namespace dax {
namespace exec {
namespace internal {

DAX_EXEC_EXPORT void derivativeWeightsVoxel(const dax::Vector3 &pcoords,
                                            dax::Vector3 weights[8])
{
  dax::Vector3 rcoords = dax::make_Vector3(1, 1, 1) - pcoords;

  weights[0][0] = -rcoords[1]*rcoords[2];
  weights[0][1] = -rcoords[0]*rcoords[2];
  weights[0][2] = -rcoords[0]*rcoords[1];

  weights[1][0] = rcoords[1]*rcoords[2];
  weights[1][1] = -pcoords[0]*rcoords[2];
  weights[1][2] = -pcoords[0]*rcoords[1];

  weights[2][0] = pcoords[1]*rcoords[2];
  weights[2][1] = pcoords[0]*rcoords[2];
  weights[2][2] = -pcoords[0]*pcoords[1];

  weights[3][0] = -pcoords[1]*rcoords[2];
  weights[3][1] = rcoords[0]*rcoords[2];
  weights[3][2] = -rcoords[0]*pcoords[1];

  weights[4][0] = -rcoords[1]*pcoords[2];
  weights[4][1] = -rcoords[0]*pcoords[2];
  weights[4][2] = rcoords[0]*rcoords[1];

  weights[5][0] = rcoords[1]*pcoords[2];
  weights[5][1] = -pcoords[0]*pcoords[2];
  weights[5][2] = pcoords[0]*rcoords[1];

  weights[6][0] = pcoords[1]*pcoords[2];
  weights[6][1] = pcoords[0]*pcoords[2];
  weights[6][2] = pcoords[0]*pcoords[1];

  weights[7][0] = -pcoords[1]*pcoords[2];
  weights[7][1] = rcoords[0]*pcoords[2];
  weights[7][2] = rcoords[0]*pcoords[1];
}

}}}

#endif //__dax__exec__internal__DerviativeWeights_h
