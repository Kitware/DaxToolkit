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
//  Copyright 2013 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//===========================================x==================================
#ifndef __dax_cont_DispatcherGenerateKeysValues_h
#define __dax_cont_DispatcherGenerateKeysValues_h

#include <dax/Types.h>
#include <dax/cont/dispatcher/DispatcherBase.h>
#include <dax/cont/internal/DeviceAdapterTag.h>
#include <dax/exec/WorkletGenerateKeysValues.h>
#include <dax/internal/ParameterPack.h>

#include <dax/cont/dispatcher/AddVisitIndexArg.h>

namespace dax { namespace cont {


template <
  class WorkletType_,
  class OutputCountType_ = dax::cont::ArrayHandle< dax::Id >,
  class DeviceAdapterTag_ = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class DispatcherGenerateKeysValues :
  public dax::cont::dispatcher::DispatcherBase<
          DispatcherGenerateKeysValues< WorkletType_, OutputCountType_, DeviceAdapterTag_ >,
          dax::exec::WorkletGenerateKeysValues,
          WorkletType_,
          DeviceAdapterTag_ >
{

  typedef dax::cont::dispatcher::DispatcherBase< DispatcherGenerateKeysValues< WorkletType_, OutputCountType_,  DeviceAdapterTag_>,
                                                 dax::exec::WorkletGenerateKeysValues,
                                                 WorkletType_,
                                                 DeviceAdapterTag_> Superclass;
  friend class dax::cont::dispatcher::DispatcherBase< DispatcherGenerateKeysValues< WorkletType_, OutputCountType_, DeviceAdapterTag_>,
                                                 dax::exec::WorkletGenerateKeysValues,
                                                 WorkletType_,
                                                 DeviceAdapterTag_>;

public:
  typedef WorkletType_ WorkletType;
  typedef OutputCountType_ OutputCountType;
  typedef DeviceAdapterTag_ DeviceAdapterTag;

  DAX_CONT_EXPORT
  DispatcherGenerateKeysValues(OutputCountType outputCountArray):
    Superclass(WorkletType()),
    ReleaseOutputCountArray(true),
    OutputCountArray(outputCountArray)
    { }

  DAX_CONT_EXPORT
  DispatcherGenerateKeysValues(OutputCountType outputCountArray, WorkletType& work):
    Superclass( work ),
    ReleaseOutputCountArray(true),
    OutputCountArray(outputCountArray)
    { }

  DAX_CONT_EXPORT void SetReleaseOutputCountArray(bool flag){
    this->ReleaseOutputCountArray = flag;
  }

  DAX_CONT_EXPORT bool GetReleaseOutputCountArray() const {
    return this->ReleaseOutputCountArray;
  }

  DAX_CONT_EXPORT OutputCountType GetOutputCountArray() const {
    return this->OutputCountArray;
  }

  DAX_CONT_EXPORT void DoReleaseOutputCountArray() {
    this->OutputCountArray.ReleaseResourcesExecution();
  }

private:

  template<typename ParameterPackType>
  DAX_CONT_EXPORT void DoInvoke(WorkletType worklet,
                                ParameterPackType arguments)
  {
    this->InvokeGenerateKeysValues(
          worklet,
          dax::internal::ParameterPackGetArgument<1>(arguments),
          arguments);
  }


template <typename InputGrid,
          typename ParameterPackType>
DAX_CONT_EXPORT void InvokeGenerateKeysValues(
    WorkletType worklet,
    const InputGrid inputGrid,
    const ParameterPackType &arguments)
  {
  typedef dax::cont::DeviceAdapterAlgorithm<DeviceAdapterTag> Algorithm;
  typedef dax::cont::ArrayHandle<dax::Id, ArrayContainerControlTagBasic,
      DeviceAdapterTag> IdArrayHandleType;

  //do an inclusive scan of the cell count / cell mask to get the number
  //of cells in the output
  IdArrayHandleType scannedOutputCounts;
  const dax::Id numNewValues =
      Algorithm::ScanInclusive(this->GetOutputCountArray(),
                               scannedOutputCounts);

  if(this->GetReleaseOutputCountArray())
    {
    this->DoReleaseOutputCountArray();
    }

  if(numNewValues == 0)
    {
    //nothing to do
    return;
    }


  //now do the lower bounds of the cell indices so that we figure out
  IdArrayHandleType outputIndexRanges;
  Algorithm::UpperBounds(scannedOutputCounts,
                 dax::cont::make_ArrayHandleCounting(dax::Id(0),numNewValues),
                 outputIndexRanges);

  // We are done with scannedOutputCounts.
  scannedOutputCounts.ReleaseResources();

  //we need to scan the args of the generate topology worklet and determine if
  //we have the VisitIndex signature. If we do, we have to call a different
  //Invoke algorithm, which properly uploads the visitIndex information. Since
  //this information is slow to compute we don't want to always upload the
  //information, instead only compute when explicitly requested

  //The AddVisitIndexArg does all this, plus creates a derived worklet
  //from the users worklet with the visit index added to the signature.
  typedef dax::cont::dispatcher::AddVisitIndexArg<WorkletType,
    Algorithm,IdArrayHandleType> AddVisitIndexFunctor;
  typedef typename AddVisitIndexFunctor::VisitIndexArgType IndexArgType;
  typedef typename AddVisitIndexFunctor::DerivedWorkletType DerivedWorkletType;

  IndexArgType visitIndex;
  AddVisitIndexFunctor createVisitIndex;
  createVisitIndex(outputIndexRanges,visitIndex);

  DerivedWorkletType derivedWorklet(worklet);

  //we get our magic here. we need to wrap some paramemters and pass
  //them to the real dispatcher
  this->BasicInvoke( derivedWorklet,
        arguments.template Replace<1>(
            dax::cont::make_Permutation(outputIndexRanges,inputGrid,
                                        inputGrid.GetNumberOfCells()))
        .Append(visitIndex));
  }

  bool ReleaseOutputCountArray;
  OutputCountType OutputCountArray;
};

} }
#endif //__dax_cont_DispatcherGenerateKeysValues_h
