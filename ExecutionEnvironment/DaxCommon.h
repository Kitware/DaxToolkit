/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/// This file defines some common stuff.

#ifndef __DaxCommon_h
#define __DaxCommon_h

//*****************************************************************************
// Typedefs for basic types.
//*****************************************************************************
typedef float DaxScalar;
typedef float3 DaxVector3;
typedef float4 DaxVector4;
typedef int DaxId;

//*****************************************************************************
// Cell Types
//*****************************************************************************
enum DaxCellType
{
  EMPTY_CELL       = 0,
  VERTEX           = 1,
  POLY_VERTEX      = 2,
  LINE             = 3,
  POLY_LINE        = 4,
  TRIANGLE         = 5,
  TRIANGLE_STRIP   = 6,
  POLYGON          = 7,
  PIXEL            = 8,
  QUAD             = 9,
  TETRA            = 10,
  VOXEL            = 11,
  HEXAHEDRON       = 12,
  WEDGE            = 13,
  PYRAMID          = 14,
  PENTAGONAL_PRISM = 15,
  HEXAGONAL_PRISM  = 16,
};


#define make_DaxVector3 make_float3
#define make_DaxVector4 make_float4

#define SUPERCLASS(__name__) \
  typedef __name__ Superclass

#endif
