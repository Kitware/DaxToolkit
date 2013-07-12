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
#ifndef __dax_cont_scheduling_AddReductionCountArg_h
#define __dax_cont_scheduling_AddReductionCountArg_h

#include <dax/Types.h>

#include <dax/cont/arg/Field.h>
#include <dax/cont/sig/Arg.h>
#include <dax/cont/sig/ReductionCount.h>

// For DerivedWorklet. (TODO: Probably should but this somewhere more generic.)
#include <dax/exec/internal/kernel/VisitIndexWorklets.h>

//needed to parse signature arguments to determine what implicit
//args we need to upload to execution
#include <dax/internal/WorkletSignatureFunctions.h>


namespace dax { namespace cont { namespace scheduling {

template<class WorkletType>
class AddReductionCountArg
{
  typedef dax::internal::ReplaceAndExtendSignatures<
              WorkletType,
              dax::cont::sig::ReductionCount,
              dax::cont::sig::internal::ReductionCountMetaFunc,
              dax::cont::arg::Field>  ModifiedWorkletSignatures;

  typedef typename ModifiedWorkletSignatures::found ReductionCountFound;


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

};

template<class WorkletType>
class AddReductionOffsetArg
{
  typedef dax::internal::ReplaceAndExtendSignatures<
              WorkletType,
              dax::cont::sig::ReductionOffset,
              dax::cont::sig::internal::ReductionOffsetMetaFunc,
              dax::cont::arg::Field>  ModifiedWorkletSignatures;

  typedef typename ModifiedWorkletSignatures::found ReductionOffsetFound;


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

};

template<class WorkletType>
class AddReductionIndexPortalArg
{
  typedef dax::internal::ReplaceAndExtendSignatures<
              WorkletType,
              dax::cont::sig::ReductionIndexPortal,
              dax::cont::sig::internal::ReductionIndexPortalMetaFunc,
              dax::cont::arg::Field>  ModifiedWorkletSignatures;

  typedef typename ModifiedWorkletSignatures::found ReductionIndexPortalFound;


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

};
} } } //dax::cont::scheduling
#endif //__dax_cont_scheduling_AddReductionCountArg_h
