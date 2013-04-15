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
//===========================================x==================================
#ifndef __dax_cont_scheduling_AddVisitIndexArg_h
#define __dax_cont_scheduling_AddVisitIndexArg_h

#include <dax/Types.h>

#include <dax/cont/arg/Field.h>
#include <dax/cont/sig/Arg.h>
#include <dax/cont/sig/VisitIndex.h>
#include <dax/exec/internal/kernel/VisitIndexWorklets.h>

//needed to parse signature arguments to determine what implicit
//args we need to upload to execution
#include <dax/internal/WorkletSignatureFunctions.h>


namespace dax { namespace cont { namespace scheduling {

namespace internal
{
  //This is used to generate the implict VisitIndex values.
//If GenerateVisitIndex is equal to a boost false type, we
template<bool GenerateVisitIndex,
         class Algorithm,
         typename HandleType>
struct MakeVisitIndexControlType
{
  typedef HandleType Type;

  template<class Scheduler, typename OtherHandleType>
  void operator()(Scheduler &scheduler,
                  const OtherHandleType& inputCellIds, Type& visitIndices) const
  {
    //To determine the number of times we have already visited
    //the current input cell, we take the lower bounds of the
    //input cell id array. The resulting number subtracted from the WorkId
    //gives us the number of times we have visited that cell
    visitIndices.PrepareForOutput(inputCellIds.GetNumberOfValues());
    Algorithm::LowerBounds(inputCellIds, inputCellIds, visitIndices);
    scheduler.Invoke(dax::exec::internal::kernel::ComputeVisitIndex(),
                     visitIndices);
  }
};

template<class Algorithm, typename HandleType>
struct MakeVisitIndexControlType<false,Algorithm,HandleType>
{
  typedef dax::Id Type;

  template<class Scheduler, typename OtherHandleType>
  void operator()(Scheduler &daxNotUsed(scheduler),
                  const OtherHandleType& daxNotUsed(handle), Type& value) const
  {
    //The visitIndex is not requested, so we fill in the control side argument
    //with a integer value that will be parsed by the bindings code, but
    //won't be uploaded to the execution env.
    value = 0;
  }
};
}

template<class WorkletType, class Algorithm, class IdArrayHandleType>
class AddVisitIndexArg
{
  typedef dax::internal::ReplaceAndExtendSignatures<
              WorkletType,
              dax::cont::sig::VisitIndex,
              dax::cont::sig::internal::VisitIndexMetaFunc,
              dax::cont::arg::Field>  ModifiedWorkletSignatures;

  typedef typename ModifiedWorkletSignatures::found VisitIndexFound;

  typedef internal::MakeVisitIndexControlType<VisitIndexFound::value,
              Algorithm,
              IdArrayHandleType> VisitContFunction;


  //now that we have index generated, we have to build the new worklet
  //that has the updated signature
  typedef typename dax::internal::BuildSignature<
            typename ModifiedWorkletSignatures::ControlSignature>::type NewContSig;
  typedef typename dax::internal::BuildSignature<
            typename ModifiedWorkletSignatures::ExecutionSignature>::type NewExecSig;

public:

  //make sure to pass the user worklet down to the new derived worklet, so
  //that we get any member variable values that they have set
  typedef dax::exec::internal::kernel::DerivedWorklet<WorkletType,
            NewContSig,NewExecSig> DerivedWorkletType;

  //generate the visitIndex, if we don't need it we will create a dummy
  //control side signature argument that is just a constant value dax::Id
  typedef typename VisitContFunction::Type VisitIndexArgType;


  template<class Scheduler, class HandleType>
  void operator()(Scheduler& scheduler, const HandleType& cellRange, VisitIndexArgType& visitIndex)
    {
    VisitContFunction()(scheduler,cellRange,visitIndex);
    }

};

} } } //dax::cont::scheduling
#endif //__dax_cont_scheduling_AddVisitIndexArg_h
