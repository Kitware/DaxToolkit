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

#ifndef __dax_worklet_internal_MarchingTetrahedraTable_h
#define  __dax_worklet_internal_MarchingTetrahedraTable_h

#include <dax/internal/ExportMacros.h>

namespace dax {
namespace worklet{
namespace internal{
namespace marchingtetrahedra{

//-------------------------------------------------------------------- numFaces
DAX_EXEC_CONSTANT_EXPORT const unsigned char NumFaces[16] =
{ 0, 1, 1, 2,
  1, 2, 2, 1,
  1, 2, 2, 1,
  2, 1, 1, 0};

//-------------------------------------------------------------------- triTable
DAX_EXEC_CONSTANT_EXPORT const unsigned char TriTable[16][8] =
{
    {255, 255, 255, 255, 255, 255, 255, 255},
    {0, 3, 2, 255, 255, 255, 255, 255},
    {0, 1, 4, 255, 255, 255, 255, 255},
    {1, 4, 3, 1, 3, 2, 255, 255},
    {1, 2, 5, 255, 255, 255, 255, 255},
    {0, 3, 1, 1, 3, 5, 255, 255},
    {2, 5, 4, 0, 2, 4, 255, 255},
    {3, 5, 4, 255, 255, 255, 255, 255},
    {3, 4, 5, 255, 255, 255, 255, 255},
    {2, 0, 4, 2, 4, 5, 255, 255},
    {0, 1, 5, 0, 5, 3, 255, 255},
    {1, 5, 2, 255, 255, 255, 255, 255},
    {3, 4, 1, 3, 1, 2, 255, 255},
    {0, 4, 1, 255, 255, 255, 255, 255},
    {0, 2, 3, 255, 255, 255, 255, 255},
    {255, 255, 255, 255, 255, 255, 255, 255}
};

}
}
}
} // namespace dax::worklet::internal::marchingtetrahedra

#endif //__dax_worklet_internal_MarchingTetrahedraTable_h
