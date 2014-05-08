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
#ifndef __dax_cont_DispatcherReduceKeysValues_h
#define __dax_cont_DispatcherReduceKeysValues_h

#include <dax/Types.h>
#include <dax/cont/dispatcher/DispatcherBase.h>
#include <dax/cont/internal/DeviceAdapterTag.h>
#include <dax/exec/WorkletReduceKeysValues.h>
#include <dax/internal/ParameterPack.h>

#include <dax/cont/dispatcher/AddReduceKeysArgs.h>

#include <dax/exec/internal/kernel/GenerateWorklets.h>


namespace dax { namespace cont {


template <
  class WorkletType_,
  class KeysHandleType_ = dax::cont::ArrayHandle< dax::Id >,
  class DeviceAdapterTag_ = DAX_DEFAULT_DEVICE_ADAPTER_TAG>
class DispatcherReduceKeysValues :
  public dax::cont::dispatcher::DispatcherBase<
          DispatcherReduceKeysValues< WorkletType_, KeysHandleType_, DeviceAdapterTag_ >,
          dax::exec::WorkletReduceKeysValues,
          WorkletType_,
          DeviceAdapterTag_ >
{

  typedef dax::cont::dispatcher::DispatcherBase< DispatcherReduceKeysValues< WorkletType_, KeysHandleType_,  DeviceAdapterTag_>,
                                                 dax::exec::WorkletReduceKeysValues,
                                                 WorkletType_,
                                                 DeviceAdapterTag_> Superclass;
  friend class dax::cont::dispatcher::DispatcherBase< DispatcherReduceKeysValues< WorkletType_, KeysHandleType_, DeviceAdapterTag_>,
                                                 dax::exec::WorkletReduceKeysValues,
                                                 WorkletType_,
                                                 DeviceAdapterTag_>;

public:
  typedef WorkletType_ WorkletType;
  typedef KeysHandleType_ KeysHandleType;
  typedef DeviceAdapterTag_ DeviceAdapterTag;

  typedef dax::cont::ArrayHandle<dax::Id,
                                 dax::cont::ArrayContainerControlTagBasic,
                                 DeviceAdapterTag> ReductionMapType;

  DAX_CONT_EXPORT
  DispatcherReduceKeysValues(const KeysHandleType &keys):
    Superclass(WorkletType()),
    Keys(keys),
    ReductionKeys(),
    ReductionMapValid(false),
    ReleaseKeys(true),
    ReleaseReductionMap(true),
    ReductionCounts(),
    ReductionIndices(),
    ReductionOffsets()
    { }

  DAX_CONT_EXPORT
  DispatcherReduceKeysValues(const KeysHandleType &keys,
                             const WorkletType& work):
    Superclass(work),
    Keys(keys),
    ReductionKeys(),
    ReductionMapValid(false),
    ReleaseKeys(true),
    ReleaseReductionMap(true),
    ReductionCounts(),
    ReductionIndices(),
    ReductionOffsets()
    { }

  DAX_CONT_EXPORT void SetReleaseKeys(bool flag){ this->ReleaseKeys = flag; }
  DAX_CONT_EXPORT bool GetReleaseKeys() const { return ReleaseKeys; }

  DAX_CONT_EXPORT KeysHandleType GetKeys() const { return this->Keys; }
  DAX_CONT_EXPORT void DoReleaseKeys()
    { this->Keys.ReleaseResourcesExecution(); }

  DAX_CONT_EXPORT void SetReleaseReductionMap(bool flag)
    { this->ReleaseReductionMap = flag; }
  DAX_CONT_EXPORT bool GetReleaseReductionMap() const
    { return ReleaseReductionMap; }

 DAX_CONT_EXPORT
  void DoReleaseReductionMap() {
    this->ReductionCounts.ReleaseResourcesExecution();
    this->ReductionOffsets.ReleaseResourcesExecution();
    this->ReductionIndices.ReleaseResourcesExecution();
    this->ReductionMapValid = false;
  }


private:

  template<typename ParameterPackType>
  DAX_CONT_EXPORT void DoInvoke(WorkletType worklet,
                                ParameterPackType arguments)
  {
    typedef dax::cont::DeviceAdapterAlgorithm<DeviceAdapterTag> Algorithm;

    //Get a map from output indices to input groups.
    this->BuildReductionMap();
    if (this->GetReleaseKeys())
      {
      this->DoReleaseKeys();
      }

    // We need to scan the args of the reduce keys/values worklet and determine
    // if we have the ReductionCount/Offset/Index signature.  The control signature needs to
    // be modified to add this array to the arguments and the execution
    // signature has to be modified to ensure that the ReductionCount signature
    // points to the appropriate array.  The AddReduceKeysArgs does all this.
    typedef typename dax::cont::dispatcher::AddReduceKeysArgs<
                  WorkletType>::DerivedWorkletType DerivedWorkletType;

    //we get our magic here. we need to wrap some parameters and pass
    //them to the real dispatcher
    DerivedWorkletType derivedWorklet(worklet);
    this->BasicInvoke(derivedWorklet,
                      arguments.Append(this->ReductionCounts)
                      .Append(this->ReductionOffsets)
                      .Append(this->ReductionIndices)
                      );

    if(this->GetReleaseReductionMap())
      {
      this->DoReleaseReductionMap();
      }
  }

  /// Builds a map from output indices to input indices that describes how
  /// many values are to be reduced for an entry and at what indices those
  /// values are.  See GetReductionCounts, GetReductionOffsets, and
  /// GetReductionIndices.
  DAX_CONT_EXPORT void BuildReductionMap()
  {
    typedef dax::cont::DeviceAdapterAlgorithm<DeviceAdapterTag>
        Algorithms;

    if (this->ReductionMapValid) { return; } // Nothing to do.

    // Make a copy of the keys.  (Our first step is sort, which is in place.)
    dax::cont::ArrayHandle<
        typename KeysHandleType::ValueType,
        dax::cont::ArrayContainerControlTagBasic,
        DeviceAdapterTag> sortedKeys;
    Algorithms::Copy(this->Keys, sortedKeys);

    // Initialize the indices using a counting array. After they are sorted as
    // values, they will point to the original index. Using a counting array
    // handle to initialize. You could also use a simple functor and a
    // schedule, but this is fewer lines and is probably about the same
    // runtime.
    dax::cont::ArrayHandleCounting<dax::Id, DeviceAdapterTag>
        countingArray(0, this->Keys.GetNumberOfValues());
    Algorithms::Copy(countingArray, this->ReductionIndices);

    Algorithms::SortByKey(sortedKeys, this->ReductionIndices);

    // Unique keys represents the output entries.
    Algorithms::Copy(sortedKeys, this->ReductionKeys);
    Algorithms::Unique(this->ReductionKeys);

    // Find the index of each unique key in the sorted list to get the offsets
    // into the ReductionIndices array.
    Algorithms::LowerBounds(sortedKeys, this->ReductionKeys, this->ReductionOffsets);

    //Find the number of values corresponding to each unique key.
    dax::Id numUniqueKeys = this->ReductionKeys.GetNumberOfValues();

    typedef dax::exec::internal::kernel::Offset2CountFunctor<
                        ReductionMapType> OffsetFunctorType;
    OffsetFunctorType offset2Count( this->ReductionOffsets.PrepareForInput(),
                        this->ReductionCounts.PrepareForOutput(numUniqueKeys),
                        numUniqueKeys-1,
                        this->ReductionIndices.GetNumberOfValues());
    Algorithms::Schedule(offset2Count, numUniqueKeys);

    this->ReductionMapValid = true;
  }



  KeysHandleType Keys;
  KeysHandleType ReductionKeys;

  bool ReductionMapValid;
  bool ReleaseKeys;
  bool ReleaseReductionMap;

  ReductionMapType ReductionCounts;
  ReductionMapType ReductionIndices;
  ReductionMapType ReductionOffsets;

};

} }

#endif //__dax_cont_DispatcherReduceKeysValues_h
