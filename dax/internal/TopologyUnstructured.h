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
#ifndef __dax__internal__Unstructured_h
#define __dax__internal__Unstructured_h

#include <dax/Types.h>
#include <dax/internal/GridTopologys.h>
#include <dax/internal/DataArray.h>

namespace dax {
namespace internal {


template< typename T>
struct TopologyUnstructured
{
  typedef T CellType;

  TopologyUnstructured()
    {
    }

  TopologyUnstructured(const dax::internal::DataArray<dax::Vector3>&points,
                       const dax::internal::DataArray<dax::Id>& topology):
    Points(points),
    Topology(topology)
    {
    }

  dax::internal::DataArray<dax::Vector3> Points;
  dax::internal::DataArray<dax::Id> Topology;
};

/// Returns the number of cells in a unstructured grid.
template<typename T>
DAX_EXEC_CONT_EXPORT
dax::Id numberOfCells(const TopologyUnstructured<T> &GridTopology)
{
  typedef typename TopologyUnstructured<T>::CellType CellType;
  return GridTopology.Topology.GetNumberOfEntries()/CellType::NUM_POINTS;
}

/// Returns the number of points in a unstructured grid.
template<typename T>
DAX_EXEC_CONT_EXPORT
dax::Id numberOfPoints(const TopologyUnstructured<T> &GridTopology)
{
  return GridTopology.Points.GetNumberOfEntries();
}

/// Returns the point position in a structured grid for a given index
/// which is represented by /c pointIndex
template<typename T>
DAX_EXEC_CONT_EXPORT
dax::Vector3 pointCoordiantes(const TopologyUnstructured<T> &grid,
                              dax::Id pointIndex)
{
  return grid.Points.GetValue(pointIndex);
}


} //internal
} //dax

#endif // __dax__internal__UnstructuredGrid_h
