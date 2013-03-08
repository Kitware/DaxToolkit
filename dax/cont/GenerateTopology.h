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

#ifndef __dax_cont_GenerateTopology_h
#define __dax_cont_GenerateTopology_h

#include <boost/type_traits/is_base_of.hpp>

#include <dax/Types.h>

#include <dax/exec/internal/ErrorMessageBuffer.h>
#include <dax/exec/internal/GridTopologies.h>
#include <dax/exec/WorkletGenerateTopology.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>

namespace dax {
namespace cont {

//we need a base class with no template parameters so that the scheduling triats
//infastructure can properly determine when a worklet is derived from
//GenerateTopology
namespace internal {
  class GenerateTopologyBase {};
}

/// GenerateTopology is the control enviorment representation of a
/// an algorithm that takes a classification of a given input topology
/// and will generate a new topology, but doesn't create new cells
template<
    class WorkletType_,
    class DeviceAdapterTag= DAX_DEFAULT_DEVICE_ADAPTER_TAG,
    class ClassifyContainerTag = ArrayContainerControlTagBasic>

class GenerateTopology :
    public dax::cont::internal::GenerateTopologyBase
{
  //verify that the worklet base of this class matches the scheduler
  //we are going to run on. Named after how to fix the problem with
  //using a worklet that doesn't inherit from WorkletGenerateTopology
  typedef typename boost::is_base_of<
            dax::exec::WorkletGenerateTopology,
            WorkletType_ > Worklet_Should_Inherit_From_WorkletGenerateTopology;
public:
  typedef WorkletType_ WorkletType;
  //classify type is the type
  typedef dax::Id ClassifyType;

  //mask type is the internal
  typedef dax::Id MaskType;

  typedef dax::cont::ArrayHandle< MaskType,
          ArrayContainerControlTagBasic, DeviceAdapterTag> PointMaskType;

  typedef dax::cont::ArrayHandle< ClassifyType,
          ClassifyContainerTag, DeviceAdapterTag> ClassifyResultType;

  GenerateTopology(ClassifyResultType classification):
    RemoveDuplicatePoints(true),
    ReleaseClassification(true),
    Classification(classification),
    PointMask(),
    Worklet()
    {
    BOOST_MPL_ASSERT((Worklet_Should_Inherit_From_WorkletGenerateTopology));
    }

  GenerateTopology(ClassifyResultType classification, WorkletType& work):
    RemoveDuplicatePoints(true),
    ReleaseClassification(true),
    Classification(classification),
    PointMask(),
    Worklet(work)
    {
    BOOST_MPL_ASSERT((Worklet_Should_Inherit_From_WorkletGenerateTopology));
    }

  template<typename T, typename Container1, typename Container2>
  bool CompactPointField(
      const dax::cont::ArrayHandle<T,Container1,DeviceAdapterTag>& input,
      dax::cont::ArrayHandle<T,Container2,DeviceAdapterTag>& output)
    {
    if(this->GetRemoveDuplicatePoints())
      {
      dax::cont::internal::DeviceAdapterAlgorithm<DeviceAdapterTag>::
          StreamCompact(input, this->PointMask, output);

      return true;
      }
    return false;
    }

  void SetReleaseClassification(bool b){ ReleaseClassification = b; }
  bool GetReleaseClassification() const { return ReleaseClassification; }

  ClassifyResultType GetClassification() const { return Classification; }
  void DoReleaseClassification() { Classification.ReleaseResourcesExecution(); }

  PointMaskType GetPointMask() { return PointMask; }
  const PointMaskType GetPointMask() const { return PointMask; }


  void SetRemoveDuplicatePoints(bool b){ RemoveDuplicatePoints = b; }
  bool GetRemoveDuplicatePoints() const { return RemoveDuplicatePoints; }

  WorkletType GetWorklet() const {return Worklet; }

private:
  bool RemoveDuplicatePoints;
  bool ReleaseClassification;
  ClassifyResultType Classification;
  PointMaskType PointMask;
  WorkletType Worklet;

};

} } //namespace dax::cont

#endif // __dax_cont_GenerateTopology_h
