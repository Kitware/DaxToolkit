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
#ifndef __dax_CellTraits_h
#define __dax_CellTraits_h

#include <dax/CellTag.h>
#include <dax/Types.h>

namespace dax {

/// The templated CellTraits struct provides the basic high level information
/// about cells (like the number of vertices in the cell or its
/// dimensionality).
///
template<class CellTag>
struct CellTraits
#ifdef DAX_DOXYGEN_ONLY
{
  /// Constant containing the number of vertices per cell.
  ///
  const static dax::Id NUM_POINTS = 4;

  /// This defines the topological dimensions of the class. 3 for polyhedra,
  /// 2 for polygons, 1 for lines, 0 for points.
  ///
  const static dax::Id TOPOLOGICAL_DIMENSIONS = 3;
};
#else // DAX_DOXYGEN_ONLY
    ;
#endif // DAX_DOXYGEN_ONLY

//-----------------------------------------------------------------------------

template<> struct CellTraits<dax::CellTagHexahedron> {
  const static dax::Id NUM_POINTS = 8;
  const static dax::Id TOPOLOGICAL_DIMENSIONS = 3;
};

template<> struct CellTraits<dax::CellTagLine> {
  const static dax::Id NUM_POINTS = 2;
  const static dax::Id TOPOLOGICAL_DIMENSIONS = 1;
};

template<> struct CellTraits<dax::CellTagQuadrilateral> {
  const static dax::Id NUM_POINTS = 4;
  const static dax::Id TOPOLOGICAL_DIMENSIONS = 2;
};

template<> struct CellTraits<dax::CellTagTetrahedron> {
  const static dax::Id NUM_POINTS = 4;
  const static dax::Id TOPOLOGICAL_DIMENSIONS = 3;
};

template<> struct CellTraits<dax::CellTagTriangle> {
  const static dax::Id NUM_POINTS = 3;
  const static dax::Id TOPOLOGICAL_DIMENSIONS = 2;
};

template<> struct CellTraits<dax::CellTagVertex> {
  const static dax::Id NUM_POINTS = 1;
  const static dax::Id TOPOLOGICAL_DIMENSIONS = 0;
};

template<> struct CellTraits<dax::CellTagVoxel> {
  const static dax::Id NUM_POINTS = 8;
  const static dax::Id TOPOLOGICAL_DIMENSIONS = 3;
};

template<> struct CellTraits<dax::CellTagWedge> {
  const static dax::Id NUM_POINTS = 6;
  const static dax::Id TOPOLOGICAL_DIMENSIONS = 3;
};

} // namespace dax

#endif //__dax_CellTraits_h
