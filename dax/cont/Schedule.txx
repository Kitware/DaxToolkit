#if !defined(BOOST_PP_IS_ITERATING)

# if !(__cplusplus >= 201103L)
#  include <dax/internal/ParameterPackCxx03.h>
# endif // !(__cplusplus >= 201103L)

#  define BOOST_PP_ITERATION_PARAMS_1 (3, (2, 10, <dax/cont/Schedule.txx>))
#  include BOOST_PP_ITERATE()

#else // defined(BOOST_PP_IS_ITERATING)

#if _dax_pp_sizeof___T > 1
template <class WorkletType, _dax_pp_typename___T>
Schedule(WorkletType w, _dax_pp_params___(a))
  {
  this->operator()(w, _dax_pp_args___(a));
  }

//Note any changes to this method must be reflected in the
//other implementation inisde Schedule.txx
template <class WorkletType, _dax_pp_typename___T>
void operator()(WorkletType w, _dax_pp_params___(a)) const
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
  dax::cont::internal::NG_Schedule<ControlInvocationSignature>
    (w, bindings, count, DeviceAdapterTag());
  }
# endif // _dax_pp_sizeof___T > 1
#endif // defined(BOOST_PP_IS_ITERATING)
