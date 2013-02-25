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

#ifndef __dax_cont_GenerateInterpolatedCells_h
#define __dax_cont_GenerateInterpolatedCells_h

#include <boost/type_traits/is_base_of.hpp>

#include <dax/Types.h>

#include <dax/exec/internal/ErrorMessageBuffer.h>
#include <dax/exec/internal/GridTopologies.h>
#include <dax/exec/WorkletInterpolatedCell.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>

namespace dax {
namespace cont {

//we need a base class with no template parameters so that the scheduling triats
//infastructure can properly determine when a worklet is derived from
//GenerateInterpolatedCells
namespace internal {
  class GenerateInterpolatedCellsBase {};
}

/// GenerateInterpolatedCells is the control environment representation of a
/// an algorithm that takes a classification of a given input topology
/// and will generate a new coordinates and topology
template<
    class WorkletType_,
    class DeviceAdapterTag= DAX_DEFAULT_DEVICE_ADAPTER_TAG,
    class ClassifyContainerTag = ArrayContainerControlTagBasic>

class GenerateInterpolatedCells :
    public dax::cont::internal::GenerateInterpolatedCellsBase
{
  //verify that the worklet base of this class matches the scheduler
  //we are going to run on. Named after how to fix the problem with
  //using a worklet that doesn't inherit from WorkletInterpolatedCell
  typedef typename boost::is_base_of<
          dax::exec::WorkletInterpolatedCell,
          WorkletType_ > Worklet_Should_Inherit_From_WorkletGenerateCells;
public:
  typedef WorkletType_ WorkletType;




  //classify type is the type
  typedef dax::Id ClassifyType;

  typedef dax::cont::ArrayHandle< ClassifyType,
          ClassifyContainerTag, DeviceAdapterTag> ClassifyResultType;

  GenerateInterpolatedCells(ClassifyResultType classification):
    RemoveDuplicatePoints(true),
    ReleaseClassification(true),
    Classification(classification),
    Worklet()
    {
    BOOST_MPL_ASSERT((Worklet_Should_Inherit_From_WorkletGenerateCells));
    }

  GenerateInterpolatedCells(ClassifyResultType classification, WorkletType& work):
    RemoveDuplicatePoints(true),
    ReleaseClassification(true),
    Classification(classification),
    Worklet(work)
    {
    BOOST_MPL_ASSERT((Worklet_Should_Inherit_From_WorkletGenerateCells));
    }

  void SetReleaseClassification(bool b){ ReleaseClassification = b; }
  bool GetReleaseClassification() const { return ReleaseClassification; }

  ClassifyResultType GetClassification() const { return Classification; }
  void DoReleaseClassification() { Classification.ReleaseResourcesExecution(); }

  void SetRemoveDuplicatePoints(bool b){ RemoveDuplicatePoints = b; }
  bool GetRemoveDuplicatePoints() const { return RemoveDuplicatePoints; }

  WorkletType GetWorklet() const {return Worklet; }

private:
  bool RemoveDuplicatePoints;
  bool ReleaseClassification;
  ClassifyResultType Classification;
  WorkletType Worklet;

};

} } //namespace dax::cont

#endif // __dax_cont_GenerateInterpolatedCells_h
