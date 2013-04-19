
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

#ifndef __Tetrahedralize_worklet_
#define __Tetrahedralize_worklet_

#include <dax/exec/CellField.h>
#include <dax/exec/CellVertices.h>
#include <dax/exec/VectorOperations.h>
#include <dax/exec/WorkletMapCell.h>
#include <dax/exec/WorkletGenerateTopology.h>
#include <dax/VectorTraits.h>

//to make the lookup table faster on cuda machines, and smaller overall
//I am going to declare as a constant lookup table

namespace dax { namespace worklet {

namespace detail{
DAX_EXEC_CONSTANT_EXPORT const unsigned char PossibleTetraSplitCases[10][4] =
{
{1,3,6,2},
{0,1,3,4},
{1,4,5,6},
{3,6,7,4},
{1,4,6,3},
{2,1,5,0},
{0,2,3,7},
{2,5,6,7},
{0,7,4,5},
{0,2,7,5}
};
}

//currently only supports voxel to tetra's
class Tetrahedralize : public dax::exec::WorkletGenerateTopology
{
public:
  typedef void ControlSignature(Topology, Topology(Out));
  typedef void ExecutionSignature(Vertices(_1),Vertices(_2), WorkId, VisitIndex);

  DAX_EXEC_EXPORT
  void operator()(const dax::exec::CellVertices<dax::CellTagVoxel> &inVertices,
                  dax::exec::CellVertices<dax::CellTagTetrahedron> &outVertices,
                  const dax::Id inputCellId,
                  const dax::Id visitIndex) const
  {
  //instead of swtiching on id % 2, we take it as zero or one and multiple
  //by 5 to get the correct index in our lookup table
  dax::Id lookupTablePos = visitIndex + (  5 * inputCellId % 2 );

  outVertices[0] = inVertices[ detail::PossibleTetraSplitCases[lookupTablePos][0] ];
  outVertices[1] = inVertices[ detail::PossibleTetraSplitCases[lookupTablePos][1] ];
  outVertices[2] = inVertices[ detail::PossibleTetraSplitCases[lookupTablePos][2] ];
  outVertices[3] = inVertices[ detail::PossibleTetraSplitCases[lookupTablePos][3] ];
  }
};

}}
#endif
