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
#if !defined(BOOST_PP_IS_ITERATING)
# define BOOST_PP_ITERATION_PARAMS_1 (3, (1, 10, <dax/cont/Schedule_Cxx03.h>))
# include BOOST_PP_ITERATE()

#include <dax/cont/internal/Schedule_GenerateTopology.h>
# define BOOST_PP_ITERATION_PARAMS_1 (3, (1, 10, <dax/cont/internal/Schedule_GenerateTopology.h>))
# include BOOST_PP_ITERATE()

#else // defined(BOOST_PP_IS_ITERATING)
# if _dax_pp_sizeof___T > 0
// Note any changes to this method must be reflected in the
// C++11 implementation inside Schedule.h.
template <class WorkletType, _dax_pp_typename___T>
typename boost::enable_if<boost::is_base_of<dax::exec::internal::WorkletBase,WorkletType> >::type
 operator()(WorkletType w, _dax_pp_params___(a)) const
  {
  // Construct the signature of the worklet invocation on the control side.
  typedef WorkletType ControlInvocationSignature(_dax_pp_T___);
  typedef typename WorkletType::WorkType WorkType;

  // Bind concrete arguments T...a to the concepts declared in the
  // worklet ControlSignature through ConceptMap specializations.
  // The concept maps also know how to make the arguments available
  // in the execution environment.
  dax::cont::internal::Bindings<ControlInvocationSignature>
    bindings(_dax_pp_args___(a));

  // Visit each bound argument to determine the count to be scheduled.
  dax::Id count=1;
  bindings.ForEach(dax::cont::detail::CollectCount<WorkType>(count));

  // Visit each bound argument to set up its representation in the
  // execution environment.
  bindings.ForEach(dax::cont::detail::CreateExecutionResources<WorkType>(count));

  // Schedule the worklet invocations in the execution environment.
  dax::cont::internal::Schedule<ControlInvocationSignature>
    (w, bindings, count, DeviceAdapterTag());
  }

# endif // _dax_pp_sizeof___T > 1
#endif // defined(BOOST_PP_IS_ITERATING)
