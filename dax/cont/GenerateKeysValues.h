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

#ifndef __dax_cont_GenerateKeysValues_h
#define __dax_cont_GenerateKeysValues_h

#include <boost/type_traits/is_base_of.hpp>

#include <dax/Types.h>

#include <dax/exec/internal/ErrorMessageBuffer.h>
#include <dax/exec/internal/GridTopologies.h>
#include <dax/exec/WorkletGenerateKeysValues.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>

namespace dax {
namespace cont {

//we need a base class with no template parameters so that the scheduling triats
//infastructure can properly determine when a worklet is derived from
//GenerateKeysValues
namespace internal {
  class GenerateKeysValuesBase {};
}

/// GenerateKeysValues is the control environment representation of an algorithm
/// that takes an input array containing in each entry the count of output
/// entries (0 or more) to produce for that cell.
///
template<
    class WorkletType_,
    class OutputCountHandleType = dax::cont::ArrayHandle< dax::Id >
    >

class GenerateKeysValues :
    public dax::cont::internal::GenerateKeysValuesBase
{
  //verify that the worklet base of this class matches the scheduler
  //we are going to run on. Named after how to fix the problem with
  //using a worklet that doesn't inherit from WorkletGenerateKeysValues
  typedef typename boost::is_base_of<
            dax::exec::WorkletGenerateKeysValues,
            WorkletType_ > Worklet_Should_Inherit_From_WorkletGenerateKeysValues;
public:
  typedef WorkletType_ WorkletType;

  typedef OutputCountHandleType OutputCountType;

  GenerateKeysValues(OutputCountType outputCountArray):
    ReleaseOutputCountArray(true),
    OutputCountArray(outputCountArray)
    {
    BOOST_MPL_ASSERT((Worklet_Should_Inherit_From_WorkletGenerateKeysValues));
    }

  GenerateKeysValues(OutputCountType outputCountArray, WorkletType& work):
    ReleaseOutputCountArray(true),
    OutputCountArray(outputCountArray),
    Worklet(work)
    {
    BOOST_MPL_ASSERT((Worklet_Should_Inherit_From_WorkletGenerateKeysValues));
    }

  void SetReleaseOutputCountArray(bool flag){
    this->ReleaseOutputCountArray = flag;
  }
  bool GetReleaseOutputCountArray() const {
    return this->ReleaseOutputCountArray;
  }

  OutputCountType GetOutputCountArray() const {
    return this->OutputCountArray;
  }
  void DoReleaseOutputCountArray() {
    OutputCountArray.ReleaseResourcesExecution();
  }

  WorkletType GetWorklet() const { return this->Worklet; }

private:
  bool ReleaseOutputCountArray;
  OutputCountType OutputCountArray;
  WorkletType Worklet;

};

} } //namespace dax::cont

#endif // __dax_cont_GenerateKeysValues_h
