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

#ifndef __dax_cont_internal_ScheduleGenerateTopology_h
#define __dax_cont_internal_ScheduleGenerateTopology_h

#include <dax/Types.h>
#include <dax/exec/Cell.h>

#include <dax/exec/internal/ErrorMessageBuffer.h>
#include <dax/exec/internal/GridTopologies.h>

#include <dax/cont/DeviceAdapter.h>

namespace dax {
namespace cont {

/// ScheduleGenerateTopology is the control enviorment representation of a
/// an algorithm that takes a classification of a given input topology
/// and will generate a new topology, but doesn't create new cells

template<
    class WorkletType,
    class DeviceAdapterTag= DAX_DEFAULT_DEVICE_ADAPTER_TAG>

class ScheduleGenerateTopology
{
public:
  //classify type is the type
  typedef dax::Id ClassifyType;

  //mask type is the internal
  typedef dax::Id MaskType;

  typedef dax::cont::ArrayHandle< MaskType,
          ArrayContainerControlTagBasic, DeviceAdapterTag> PointMaskType;

  typedef dax::cont::ArrayHandle< ClassifyType,
          ArrayContainerControlTagBasic, DeviceAdapterTag> ClassifyResultType;

  ScheduleGenerateTopology(ClassifyResultType classification):
    RemoveDuplicatePoints(true),
    ReleaseClassification(true),
    Classification(classification),
    PointMask(),
    Worklet()
    {
    }

  ScheduleGenerateTopology(ClassifyResultType classification, WorkletType& work):
    RemoveDuplicatePoints(true),
    ReleaseClassification(true),
    Classification(classification),
    PointMask(),
    Worklet(work)
    {
    }

  template<typename T, typename Container1, typename Container2>
  bool CompactPointField(
      const dax::cont::ArrayHandle<T,Container1,DeviceAdapterTag>& input,
      dax::cont::ArrayHandle<T,Container2,DeviceAdapterTag>& output)
    {
    if(this->GetRemoveDuplicatePoints())
      {
      dax::cont::internal::StreamCompact(input,
                                       this->PointMask,
                                       output,
                                       DeviceAdapterTag());

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

#endif // __dax_cont_ScheduleGenerateTopology_h
