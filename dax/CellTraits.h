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

/// dax::CellTraits::TopologyDimensionType is typedef to this with the
/// template parameter set to TOPOLOGICAL_DIMENSIONS.  See dax::CellTraits
/// for more information.
///
template<int dimension>
struct CellTopologicalDimensionsTag { };

/// Tag to identify a grid type.  Used in dax::CellTraits::GridTag.
///
struct GridTagUniform { };
struct GridTagUnstructured { };

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
  const static int NUM_VERTICES = 8;

  /// This defines the topological dimensions of the class. 3 for polyhedra,
  /// 2 for polygons, 1 for lines, 0 for points.
  ///
  const static int TOPOLOGICAL_DIMENSIONS = 3;

  /// This tag is typedef'ed to
  /// dax::CellTopologicalDimensionsTag<TOPOLOGICAL_DIMENSIONS>. This provides
  /// a convienient way to overload a function based on topological dimensions
  /// (which is usually more efficient than conditionals).
  ///
  typedef dax::CellTopologicalDimensionsTag<TOPOLOGICAL_DIMENSIONS>
      TopologicalDimensionsTag;

  /// Tag to identify what type of grid this cell type belongs to.
  ///
  typedef dax::GridTagUnstructured GridTag;

  /// Describes the tag for an unstructured grid cell for which this type is
  /// topologically equivalent. If CellTag is a cell type that is used in an
  /// unstructured grid (i.e., GridTag is dax::GridTagUnstructured), then this
  /// type is the same as CellTag. Otherwise provides a similar type of cell in
  /// an unstructured grid. For example, for a CellTagVoxel, this will be cast
  /// as CellTagHexahedra. You cannot directly cast between the types of cells,
  /// but Dax functions that operate on CanonicalCellTag should work on
  /// CellTag.
  ///
  typedef dax::CellTagHexahedron CanonicalCellTag;

};
#else // DAX_DOXYGEN_ONLY
    ;
#endif // DAX_DOXYGEN_ONLY

//-----------------------------------------------------------------------------

template<> struct CellTraits<dax::CellTagHexahedron> {
  const static int NUM_VERTICES = 8;
  const static int TOPOLOGICAL_DIMENSIONS = 3;
  typedef dax::CellTopologicalDimensionsTag<3> TopologicalDimensionsTag;
  typedef dax::GridTagUnstructured GridTag;
  typedef dax::CellTagHexahedron CanonicalCellTag;
};

template<> struct CellTraits<dax::CellTagLine> {
  const static int NUM_VERTICES = 2;
  const static int TOPOLOGICAL_DIMENSIONS = 1;
  typedef dax::CellTopologicalDimensionsTag<1> TopologicalDimensionsTag;
  typedef dax::GridTagUnstructured GridTag;
  typedef dax::CellTagLine CanonicalCellTag;
};

template<> struct CellTraits<dax::CellTagQuadrilateral> {
  const static int NUM_VERTICES = 4;
  const static int TOPOLOGICAL_DIMENSIONS = 2;
  typedef dax::CellTopologicalDimensionsTag<2> TopologicalDimensionsTag;
  typedef dax::GridTagUnstructured GridTag;
  typedef dax::CellTagQuadrilateral CanonicalCellTag;
};

template<> struct CellTraits<dax::CellTagTetrahedron> {
  const static int NUM_VERTICES = 4;
  const static int TOPOLOGICAL_DIMENSIONS = 3;
  typedef dax::CellTopologicalDimensionsTag<3> TopologicalDimensionsTag;
  typedef dax::GridTagUnstructured GridTag;
  typedef dax::CellTagTetrahedron CanonicalCellTag;
};

template<> struct CellTraits<dax::CellTagTriangle> {
  const static int NUM_VERTICES = 3;
  const static int TOPOLOGICAL_DIMENSIONS = 2;
  typedef dax::CellTopologicalDimensionsTag<2> TopologicalDimensionsTag;
  typedef dax::GridTagUnstructured GridTag;
  typedef dax::CellTagTriangle CanonicalCellTag;
};

template<> struct CellTraits<dax::CellTagVertex> {
  const static int NUM_VERTICES = 1;
  const static int TOPOLOGICAL_DIMENSIONS = 0;
  typedef dax::CellTopologicalDimensionsTag<0> TopologicalDimensionsTag;
  typedef dax::GridTagUnstructured GridTag;
  typedef dax::CellTagVertex CanonicalCellTag;
};

template<> struct CellTraits<dax::CellTagVoxel> {
  const static int NUM_VERTICES = 8;
  const static int TOPOLOGICAL_DIMENSIONS = 3;
  typedef dax::CellTopologicalDimensionsTag<3> TopologicalDimensionsTag;
  typedef dax::GridTagUniform GridTag;
  typedef dax::CellTagHexahedron CanonicalCellTag;
};

template<> struct CellTraits<dax::CellTagWedge> {
  const static int NUM_VERTICES = 6;
  const static int TOPOLOGICAL_DIMENSIONS = 3;
  typedef dax::CellTopologicalDimensionsTag<3> TopologicalDimensionsTag;
  typedef dax::GridTagUnstructured GridTag;
  typedef dax::CellTagWedge CanonicalCellTag;
};

} // namespace dax

#endif //__dax_CellTraits_h
