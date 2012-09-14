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

#ifndef __dax_cuda_cont_worklet_CountPointUsage_h
#define __dax_cuda_cont_worklet_CountPointUsage_h

// TODO: This should be auto-generated.

#include <dax/worklet/Threshold.worklet>

#include <boost/shared_ptr.hpp>

#include <dax/Types.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>

#include <dax/cont/ScheduleGenerateTopology.h>


namespace dax {
namespace cont {
namespace worklet {

template<class InGridType,
         class OutGridType,
         typename ValueType,
         class Container1,
         class Container2,
         class Adapter>
inline void Threshold(
    const InGridType &inGrid,
    OutGridType &outGeom,
    ValueType thresholdMin,
    ValueType thresholdMax,
    const dax::cont::ArrayHandle<ValueType,Container1,Adapter> &thresholdHandle,
    dax::cont::ArrayHandle<ValueType,Container2,Adapter> &thresholdResult)
{

  typedef dax::cont::ScheduleGenerateTopology<Adapter> ScheduleGT;
  typedef typename ScheduleGT::ClassifyResultType  ClassifyResultType;
  typedef dax::worklet::ThresholdClassify<ValueType> ThresholdClassifyType;

  ClassifyResultType classification;
  dax::cont::Schedule<Adapter>(
                    ThresholdClassifyType(thresholdMin,thresholdMax),
                    inGrid, thresholdHandle, classification);

  ScheduleGT resolveTopology(classification);
  //remove classification resource from execution for more space
  resolveTopology.SetReleaseClassification(true);
  //resolve duplicates points
  resolveTopology.SetRemoveDuplicatePoints(true);
  resolveTopology.CompactTopology(dax::worklet::ThresholdTopology(),inGrid,outGeom);
  resolveTopology.CompactPointField(thresholdHandle,thresholdResult);
}

}
}
} //dax::cuda::cont::worklet

#endif //__dax_cuda_cont_worklet_CountPointUsage_h
