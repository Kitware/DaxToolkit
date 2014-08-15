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
#ifndef __dax_cont_dispatcher_AddReduceKeysArgs_h
#define __dax_cont_dispatcher_AddReduceKeysArgs_h

#include <dax/Types.h>

#include <dax/cont/arg/Field.h>
#include <dax/cont/sig/Arg.h>
#include <dax/cont/sig/ReductionCount.h>


// For DerivedWorklet. (TODO: Probably should but this somewhere more generic.)
#include <dax/exec/internal/kernel/VisitIndexWorklets.h>

//needed to parse signature arguments to determine what implicit
//args we need to upload to execution
#include <dax/internal/WorkletSignatureFunctions.h>


namespace dax { namespace cont { namespace dispatcher {


namespace internal{
//Control side only structure
template<typename Functor>
struct AddReductionSignatures
{
private:


public:

    typedef dax::internal::detail::ConvertToBoost<Functor> BoostTypes;

    //TODO: un-kludge this part and its related uses below.
    typedef typename BoostTypes::ContSize ReductionCountPlaceHolderPos;

    typedef typename dax::cont::sig::internal::ReductionCountMetaFunc::template
      apply<ReductionCountPlaceHolderPos>::type ReductionCountReplacementArg;

    typedef dax::internal::detail::Replace<
                      typename BoostTypes::ExecutionSignature,
                      dax::cont::sig::ReductionCount,
                      ReductionCountReplacementArg> CountReplacedExecSigArg;

    typedef typename boost::mpl::size<typename CountReplacedExecSigArg::type> ReductionOffsetPlaceHolderPos;

    typedef typename dax::cont::sig::internal::ReductionOffsetMetaFunc::template
      apply<ReductionOffsetPlaceHolderPos>::type ReductionOffsetReplacementArg;


    typedef dax::internal::detail::Replace<
                        typename CountReplacedExecSigArg::type,
                        dax::cont::sig::ReductionOffset,
                        ReductionOffsetReplacementArg> OffsetReplacedExecSigArg;

    typedef typename boost::mpl::size<typename OffsetReplacedExecSigArg::type> ReductionIndexPortalPlaceHolderPos;

    typedef typename dax::cont::sig::internal::ReductionIndexPortalMetaFunc::template
      apply<ReductionIndexPortalPlaceHolderPos>::type ReductionIndexPortalReplacementArg;


    typedef dax::internal::detail::Replace<
                        typename OffsetReplacedExecSigArg::type,
                        dax::cont::sig::ReductionIndexPortal,
                        ReductionIndexPortalReplacementArg> AllReplacedExecSigArg;


  //create the struct that will return us the new control signature. We
  //always push back on the control signature, even when we didn't replace
  //anything. This makes the code easier to read, and the code that fills
  //the control signature arguments will pass a dummy argument value
    typedef typename dax::internal::detail::PushBack<
                        typename BoostTypes::ControlSignature,
                        dax::cont::arg::Field> PushBackReductionCountSig;
    typedef typename dax::internal::detail::PushBack<
                        typename PushBackReductionCountSig::type,
                        dax::cont::arg::Field> PushBackReductionOffsetSig;
    typedef typename dax::internal::detail::PushBack<
                        typename PushBackReductionOffsetSig::type,
                        dax::cont::arg::Field> PushBackAllContSig;


    //expose our new execution signature
    typedef typename AllReplacedExecSigArg::type ExecutionSignature;

    //expose the new control signature
    typedef typename PushBackAllContSig::type ControlSignature;
};
}

template<class WorkletType>
class AddReduceKeysArgs
{
  typedef  internal::AddReductionSignatures<
              WorkletType>  ModifiedWorkletSignatures;
public:

  //make sure to pass the user worklet down to the new derived worklet, so
  //that we get any member variable values that they have set
  typedef dax::exec::internal::kernel::DerivedWorklet<WorkletType,
            ModifiedWorkletSignatures> DerivedWorkletType;

};

} } } //dax::cont::dispatcher
#endif //__dax_cont_dispatcher_AddReduceKeysArgs_h
