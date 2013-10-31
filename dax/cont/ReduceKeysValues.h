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
//=============================================================================

#ifndef __dax_cont_ReduceKeysValues_h
#define __dax_cont_ReduceKeysValues_h

#include <boost/type_traits/is_base_of.hpp>

#include <dax/Types.h>

#include <dax/exec/internal/GridTopologies.h>
#include <dax/exec/internal/WorkletBase.h>
#include <dax/exec/WorkletReduceKeysValues.h>

#include <dax/cont/ArrayContainerControlBasic.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ArrayHandleCounting.h>
#include <dax/cont/DeviceAdapter.h>

#include <dax/cont/internal/DeviceAdapterAlgorithm.h>


namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template< typename ReductionMapType >
struct Offset2CountFunctor : dax::exec::internal::WorkletBase
{
  typename ReductionMapType::PortalConstExecution OffsetsPortal;
  typename ReductionMapType::PortalExecution CountsPortal;
  dax::Id MaxId;
  dax::Id OffsetEnd;

  Offset2CountFunctor(
      typename ReductionMapType::PortalConstExecution offsetsPortal,
      typename ReductionMapType::PortalExecution countsPortal,
      dax::Id maxId,
      dax::Id offsetEnd)
    : OffsetsPortal(offsetsPortal),
      CountsPortal(countsPortal),
      MaxId(maxId),
      OffsetEnd(offsetEnd) {  }

  DAX_EXEC_EXPORT
  void operator()(dax::Id index) const {
    dax::Id thisOffset = this->OffsetsPortal.Get(index);
    dax::Id nextOffset;
    if (index == this->MaxId)
      {
      nextOffset = this->OffsetEnd;
      }
    else
      {
      nextOffset = this->OffsetsPortal.Get(index+1);
      }
    this->CountsPortal.Set(index, nextOffset - thisOffset);
  }
};
}
}
}
}

namespace dax {
namespace cont {


// template<class IteratorType>
// void PrintArray(IteratorType beginIter, IteratorType endIter)
// {
//   for (IteratorType iter = beginIter; iter != endIter; iter++)
//     {
//     std::cout << " " << *iter;
//     }
//   std::cout << std::endl;
// }


//we need a base class with no template parameters so that the scheduling triats
//infastructure can properly determine when a worklet is derived from
//ReduceKeysValues
namespace internal {
  class ReduceKeysValuesBase {};
}
/// ReduceKeysValues is the control environment representation of a
/// an algorithm that takes a classification of a given input topology
/// and will generate a new coordinates and topology
template<
    class WorkletType_,
    class KeysHandleType = dax::cont::ArrayHandle< dax::Id >
    >

class ReduceKeysValues :
    public dax::cont::internal::ReduceKeysValuesBase
{
  //verify that the worklet base of this class matches the scheduler
  //we are going to run on. Named after how to fix the problem with
  //using a worklet that doesn't inherit from WorkletReducedKeysValues
  typedef typename boost::is_base_of<
          dax::exec::WorkletReduceKeysValues,
          WorkletType_ > Worklet_Should_Inherit_From_WorkletReduceKeysValues;
public:
  typedef WorkletType_ WorkletType;
  typedef typename KeysHandleType::DeviceAdapterTag DeviceAdapterTag;

  typedef KeysHandleType KeysType;

  typedef dax::cont::ArrayHandle<dax::Id,
                                 dax::cont::ArrayContainerControlTagBasic,
                                 DeviceAdapterTag> ReductionMapType;

  DAX_CONT_EXPORT
  ReduceKeysValues(const KeysType &keys):
    ReleaseKeys(true),
    Keys(keys),
    ReleaseReductionMap(false),
    ReductionCounts(),
    ReductionOffsets(),
    ReductionIndices(),
    ReductionKeys(),
    ReductionMapValid(false),
    Worklet()
    {
    BOOST_MPL_ASSERT((Worklet_Should_Inherit_From_WorkletReduceKeysValues));
    }

  DAX_CONT_EXPORT
  ReduceKeysValues(const KeysType &keys,
                   const WorkletType& work):
    ReleaseKeys(true),
    Keys(keys),
    ReleaseReductionMap(false),
    ReductionCounts(),
    ReductionOffsets(),
    ReductionIndices(),
    ReductionKeys(),
    ReductionMapValid(false),
    Worklet(work)
    {
    BOOST_MPL_ASSERT((Worklet_Should_Inherit_From_WorkletReduceKeysValues));
    }

  DAX_CONT_EXPORT
  void SetReleaseKeys(bool flag){ this->ReleaseKeys = flag; }
  DAX_CONT_EXPORT
  bool GetReleaseKeys() const { return ReleaseKeys; }

  DAX_CONT_EXPORT
  KeysType GetKeys() const { return this->Keys; }
  DAX_CONT_EXPORT
  void DoReleaseKeys() { this->Keys.ReleaseResourcesExecution(); }

  DAX_CONT_EXPORT
  void SetReleaseReductionMap(bool flag){ this->ReleaseReductionMap = flag; }
  DAX_CONT_EXPORT
  bool GetReleaseReductionMap() const { return ReleaseReductionMap; }

public:
  /// Builds a map from output indices to input indices that describes how
  /// many values are to be reduced for an entry and at what indices those
  /// values are.  See GetReductionCounts, GetReductionOffsets, and
  /// GetReductionIndices.
  DAX_CONT_EXPORT
  void BuildReductionMap()
  {
    typedef dax::cont::DeviceAdapterAlgorithm<DeviceAdapterTag>
        Algorithms;

    if (this->ReductionMapValid) { return; } // Nothing to do.

    // Make a copy of the keys.  (Our first step is sort, which is in place.)
    dax::cont::ArrayHandle<
        typename KeysType::ValueType,
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

//    std::cout << "numUniqueKeys: " << numUniqueKeys << std::endl;
//    std::cout << "ReductionCountSize: " << this->ReductionCounts.GetNumberOfValues() << std::endl;

    typedef dax::exec::internal::kernel::Offset2CountFunctor<
                        ReductionMapType> OffsetFunctorType;
    OffsetFunctorType offset2Count( this->ReductionOffsets.PrepareForInput(),
                        this->ReductionCounts.PrepareForOutput(numUniqueKeys),
                        numUniqueKeys-1,
                        this->ReductionIndices.GetNumberOfValues());
    Algorithms::Schedule(offset2Count, numUniqueKeys);

/*    std::cout << "ReductionCountSizeAfter: " << this->ReductionCounts.GetNumberOfValues() << std::endl;
    std::cout << "ReductionCounts: ";
    PrintArray(this->ReductionCounts.GetPortalConstControl().GetIteratorBegin(),
               this->ReductionCounts.GetPortalConstControl().GetIteratorEnd());
    std::cout << "ReductionOffsets: ";
    PrintArray(this->ReductionOffsets.GetPortalConstControl().GetIteratorBegin(),
               this->ReductionOffsets.GetPortalConstControl().GetIteratorEnd());
    std::cout << "ReductionValues: ";
    PrintArray(this->ReductionIndices.GetPortalConstControl().GetIteratorBegin(),
               this->ReductionIndices.GetPortalConstControl().GetIteratorEnd());*/
    this->ReductionMapValid = true;
  }

  /// \brief Stores the number of values to reduce for each key.
  ///
  /// Each index in this array cooresponds to an output value, and the entry
  /// gives the number of values combined using the reduction operation.
  ///
  DAX_CONT_EXPORT
  ReductionMapType GetReductionCounts() {
    this->BuildReductionMap();
    return this->ReductionCounts;
  }

  /// \brief Stores offsets into the ReductionIndices array.
  ///
  /// The ReductionIndices array contains groups of input values that should be
  /// reduced.  This ReductionOffsets array gives, for each output index, the
  /// offset into ReductionIndices where the group for the associated index
  /// begins.
  ///
  DAX_CONT_EXPORT
  ReductionMapType GetReductionOffsets() {
    this->BuildReductionMap();
    return this->ReductionOffsets;
  }

  /// \brief Stores the indices of groups to be reduced.
  ///
  /// The ReductionIndices array contains groups of input values that should be
  /// reduced.  Given an index i for the output array, the input values to be
  /// reduced together for this output value are given by the indices in
  /// ReductionIndices from ReductionOffsets[i] to
  /// ReductionOffsets[i]+ReductionCounts[i]-1.
  ///
  DAX_CONT_EXPORT
  ReductionMapType GetReductionIndices() {
    this->BuildReductionMap();
    return this->ReductionIndices;
  }

  /// \brief Stores the unique key for each group
  ///
  /// The ReductionKeys array contains the key for each input group to be reduced
  ///
  DAX_CONT_EXPORT
  KeysType GetReductionKeys() {
    this->BuildReductionMap();
    return this->ReductionKeys;
  }

  DAX_CONT_EXPORT
  void DoReleaseReductionMap() {
    this->ReductionCounts.ReleaseResourcesExecution();
    this->ReductionOffsets.ReleaseResourcesExecution();
    this->ReductionIndices.ReleaseResourcesExecution();
    this->ReductionMapValid = false;
  }

  DAX_CONT_EXPORT
  WorkletType GetWorklet() const {return Worklet; }

private:
  bool ReleaseKeys;
  KeysType Keys;
  bool ReleaseReductionMap;
  ReductionMapType ReductionCounts;
  ReductionMapType ReductionOffsets;
  ReductionMapType ReductionIndices;
  KeysType ReductionKeys;
  bool ReductionMapValid;
  WorkletType Worklet;

};

} } //namespace dax::cont

#endif // __dax_cont_ReduceKeysValues_h
