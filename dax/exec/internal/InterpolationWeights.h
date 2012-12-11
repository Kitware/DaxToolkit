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
#ifndef __dax__exec__internal__InterpolationWeights_h
#define __dax__exec__internal__InterpolationWeights_h

#include <dax/CellTag.h>
#include <dax/Types.h>

namespace dax {
namespace exec {
namespace internal {

DAX_EXEC_EXPORT
dax::Tuple<dax::Scalar, 8>
InterpolationWeights(const dax::Vector3 &pcoords, dax::CellTagVoxel)
{
  const dax::Vector3 rcoords = dax::make_Vector3(1, 1, 1) - pcoords;

  dax::Tuple<dax::Scalar, 8> weights;
  weights[0] = rcoords[0] * rcoords[1] * rcoords[2];
  weights[1] = pcoords[0] * rcoords[1] * rcoords[2];
  weights[2] = pcoords[0] * pcoords[1] * rcoords[2];
  weights[3] = rcoords[0] * pcoords[1] * rcoords[2];
  weights[4] = rcoords[0] * rcoords[1] * pcoords[2];
  weights[5] = pcoords[0] * rcoords[1] * pcoords[2];
  weights[6] = pcoords[0] * pcoords[1] * pcoords[2];
  weights[7] = rcoords[0] * pcoords[1] * pcoords[2];

  return weights;
}

DAX_EXEC_EXPORT
dax::Tuple<dax::Scalar, 8>
InterpolationWeights(const dax::Vector3 &pcoords, dax::CellTagHexahedron)
{
  // Hexahedron is the same as a voxel.
  return InterpolationWeights(pcoords, dax::CellTagVoxel());
}

DAX_EXEC_EXPORT
dax::Tuple<dax::Scalar, 4>
InterpolationWeights(const dax::Vector3 &pcoords, dax::CellTagTetrahedron)
{
  dax::Tuple<dax::Scalar, 4> weights;
  weights[0] = 1 - pcoords[0] - pcoords[1] - pcoords[2];
  weights[1] = pcoords[0];
  weights[2] = pcoords[1];
  weights[3] = pcoords[2];
  return weights;
}

DAX_EXEC_EXPORT
dax::Tuple<dax::Scalar, 6>
InterpolationWeights(const dax::Vector3 &pcoords, dax::CellTagWedge)
{
  dax::Tuple<dax::Scalar, 6> weights;
  weights[0] = (1 - pcoords[0] - pcoords[1]) * (1 - pcoords[2]);
  weights[1] = pcoords[1] * (1 - pcoords[2]);
  weights[2] = pcoords[0] * (1 - pcoords[2]);
  weights[3] = (1 - pcoords[0] - pcoords[1]) * pcoords[2];
  weights[4] = pcoords[1] * pcoords[2];
  weights[5] = pcoords[0] * pcoords[2];
  return weights;
}

DAX_EXEC_EXPORT
dax::Tuple<dax::Scalar, 3>
InterpolationWeights(const dax::Vector3 &pcoords, dax::CellTagTriangle)
{
  dax::Tuple<dax::Scalar, 3> weights;
  weights[0] = 1 - pcoords[0] - pcoords[1];
  weights[1] = pcoords[0];
  weights[2] = pcoords[1];
  return weights;
}

DAX_EXEC_EXPORT
dax::Tuple<dax::Scalar, 4>
InterpolationWeights(const dax::Vector3 &pcoords, dax::CellTagQuadrilateral)
{
  const dax::Vector3 rcoords = dax::make_Vector3(1, 1, 1) - pcoords;

  dax::Tuple<dax::Scalar, 4> weights;
  weights[0] = rcoords[0] * rcoords[1];
  weights[1] = pcoords[0] * rcoords[1];
  weights[2] = pcoords[0] * pcoords[1];
  weights[3] = rcoords[0] * pcoords[1];

  return weights;
}

DAX_EXEC_EXPORT
dax::Tuple<dax::Scalar, 2>
InterpolationWeights(const dax::Vector3 &pcoords, dax::CellTagLine)
{
  dax::Tuple<dax::Scalar, 2> weights;
  weights[0] = 1 - pcoords[0];
  weights[1] = pcoords[0];
  return weights;
}

DAX_EXEC_EXPORT
dax::Tuple<dax::Scalar, 1>
InterpolationWeights(const dax::Vector3 &, dax::CellTagVertex)
{
  dax::Tuple<dax::Scalar, 1> weights(dax::Scalar(1));
  return weights;
}

}}}

#endif //__dax__exec__internal__InterpolationWeights_h
