/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/// This file defines different cell types and interpolation and derivative
/// functions for those.

#ifndef __DaxCellTypes_h
#define __DaxCellTypes_h

#include "DaxCommon.h"

//*****************************************************************************
// Cell Types
//*****************************************************************************
enum DaxCellType
{
  EMPTY_CELL       = 0,
//  VERTEX           = 1,
//  POLY_VERTEX      = 2,
  LINE             = 3,
//  POLY_LINE        = 4,
//  TRIANGLE         = 5,
//  TRIANGLE_STRIP   = 6,
//  POLYGON          = 7,
//  PIXEL            = 8,
  QUAD             = 9,
//  TETRA            = 10,
  VOXEL            = 11,
//  HEXAHEDRON       = 12,
//  WEDGE            = 13,
//  PYRAMID          = 14,
//  PENTAGONAL_PRISM = 15,
//  HEXAGONAL_PRISM  = 16,
};

// Corresponds to a VOXEL.
class DaxCellVoxel
{
public:
  /// obtain the interpolation functions for a voxel.
  __device__ static void InterpolationFunctions(
    const DaxVector3& pcoords, DaxScalar functions[8])
    {
    DaxScalar rm, sm, tm;
    DaxScalar r, s, t;
    r = pcoords.x; s = pcoords.y; t = pcoords.z;
    rm = 1.0 - r;
    sm = 1.0 - s;
    tm = 1.0 - t;
    functions[0] = rm * sm * tm;
    functions[1] = r * sm * tm;
    functions[2] = rm * s * tm;
    functions[3] = r * s * tm;
    functions[4] = rm * sm * t;
    functions[5] = r * sm * t;
    functions[6] = rm * s * t;
    functions[7] = r * s * t;
    }

  /// obtain the derivatives for a voxel.
  __device__ static void InterpolationDerivs(
    const DaxVector3& pcoords, DaxScalar derivs[24])
    {
    DaxScalar rm, sm, tm;

    rm = 1. - pcoords.x;
    sm = 1. - pcoords.y;
    tm = 1. - pcoords.z;

    // r derivatives
    derivs[0] = -sm*tm;
    derivs[1] = sm*tm;
    derivs[2] = -pcoords.y*tm;
    derivs[3] = pcoords.y*tm;
    derivs[4] = -sm*pcoords.z;
    derivs[5] = sm*pcoords.z;
    derivs[6] = -pcoords.y*pcoords.z;
    derivs[7] = pcoords.y*pcoords.z;

    // s derivatives
    derivs[8] = -rm*tm;
    derivs[9] = -pcoords.x*tm;
    derivs[10] = rm*tm;
    derivs[11] = pcoords.x*tm;
    derivs[12] = -rm*pcoords.z;
    derivs[13] = -pcoords.x*pcoords.z;
    derivs[14] = rm*pcoords.z;
    derivs[15] = pcoords.x*pcoords.z;

    // t derivatives
    derivs[16] = -rm*sm;
    derivs[17] = -pcoords.x*sm;
    derivs[18] = -rm*pcoords.y;
    derivs[19] = -pcoords.x*pcoords.y;
    derivs[20] = rm*sm;
    derivs[21] = pcoords.x*sm;
    derivs[22] = rm*pcoords.y;
    derivs[23] = pcoords.x*pcoords.y;
    }
};

#endif
