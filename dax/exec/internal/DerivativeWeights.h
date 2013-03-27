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

#include <dax/CellTag.h>
#include <dax/Types.h>

namespace dax {
namespace exec {
namespace internal {

/// Returns the partial derivatives in the r, s, and t directions of the
/// interpolation weights (see InterpolationWeights.h) at the parametric
/// coordinates.
///
DAX_EXEC_EXPORT dax::Tuple<dax::Vector3,8>
DerivativeWeights(const dax::Vector3 &pcoords, dax::CellTagVoxel)
{
  dax::Tuple<dax::Vector3,8> weights;
  const dax::Vector3 rcoords = dax::make_Vector3(1, 1, 1) - pcoords;

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

  return weights;
}

/// Returns the partial derivatives in the r, s, and t directions of the
/// interpolation weights (see InterpolationWeights.h) at the parametric
/// coordinates.
///
DAX_EXEC_EXPORT dax::Tuple<dax::Vector3,8>
DerivativeWeights(const dax::Vector3 &pcoords, dax::CellTagHexahedron)
{
  // Same as voxel
  return DerivativeWeights(pcoords, dax::CellTagVoxel());
}

/// Returns the partial derivatives in the r, s, and t directions of the
/// interpolation weights (see InterpolationWeights.h) at the parametric
/// coordinates.
///
DAX_EXEC_EXPORT dax::Tuple<dax::Vector3,6>
DerivativeWeights(const dax::Vector3 &pcoords, dax::CellTagWedge)
{
  dax::Tuple<dax::Vector3,6> weights;

  weights[0][0] = -1.0 + pcoords[2];
  weights[0][1] = -1.0 + pcoords[2];
  weights[0][2] = -1.0 + pcoords[0] + pcoords[1];

  weights[1][0] = 0.0;
  weights[1][1] = 1.0 - pcoords[2];
  weights[1][2] = -pcoords[1];

  weights[2][0] = 1.0 - pcoords[2];
  weights[2][1] = 0.0;
  weights[2][2] = -pcoords[0];

  weights[3][0] = -pcoords[2];
  weights[3][1] = -pcoords[2];
  weights[3][2] = 1.0 - pcoords[0] - pcoords[1];

  weights[4][0] = 0.0;
  weights[4][1] = pcoords[2];
  weights[4][2] = pcoords[1];

  weights[5][0] = pcoords[2];
  weights[5][1] = 0.0;
  weights[5][2] = pcoords[0];

  return weights;
}

/// Returns the partial derivatives in the r and s directions of the
/// interpolation weights (see InterpolationWeights.h) at the parametric
/// coordinates.
///
DAX_EXEC_EXPORT dax::Tuple<dax::Vector3,4>
DerivativeWeightsQuadrilateral(const dax::Vector2 &pcoords)
{
  const dax::Vector2 rcoords = dax::make_Vector2(1, 1) - pcoords;
  dax::Tuple<dax::Vector3,4> weights;

  weights[0][0] = -rcoords[1];
  weights[0][1] = -rcoords[0];
  weights[0][2] = 0.0;

  weights[1][0] = rcoords[1];
  weights[1][1] = -pcoords[0];
  weights[1][2] = 0.0;

  weights[2][0] = pcoords[1];
  weights[2][1] = pcoords[0];
  weights[2][2] = 0.0;

  weights[3][0] = -pcoords[1];
  weights[3][1] = rcoords[0];
  weights[3][2] = 0.0;

  return weights;
}

/// Returns the partial derivatives in the r, s, and t directions of the
/// interpolation weights (see InterpolationWeights.h) at the parametric
/// coordinates.
///
DAX_EXEC_EXPORT dax::Tuple<dax::Vector3,4>
DerivativeWeights(const dax::Vector3 &pcoords, dax::CellTagQuadrilateral)
{
  return DerivativeWeightsQuadrilateral(dax::make_Vector2(pcoords[0],
                                                          pcoords[1]));
}

}}}

#endif //__dax__exec__internal__DerviativeWeights_h
