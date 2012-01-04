/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_VectorOperations_h
#define __dax_VectorOperations_h

#include <dax/TypeTraits.h>

namespace dax {

/// Applies a function to each entry in a vector and returns a vector of the
/// results.
///
/// \param vector is an input vector of some ambiguous type
///
/// \param functor is a function or functor class that accepts one argument of
/// the scalar type contained in the vector (specifically
/// <tt>dax::VectorTraits<VectorType>::ValueType</tt>) and returns a value of
/// the same type.
///
/// \return A vector with \c functor applied to all components of \c vector.
///
template<class VectorType, class FunctorType>
DAX_EXEC_CONT_EXPORT VectorType VectorMap(const VectorType &vector,
                                          FunctorType functor)
{
  typedef dax::VectorTraits<VectorType> Traits;
  VectorType result;
  for (int component = 0; component < Traits::NUM_COMPONENTS; component++)
    {
    Traits::SetComponent(result,
                         component,
                         functor(Traits::GetComponent(vector, component)));
    }
  return result;
}

namespace internal {

/// This hidden implementation allows us to specialize VectorReduce for single
/// component vectors.
///
template<class VectorType, class FunctorType>
DAX_EXEC_CONT_EXPORT typename dax::VectorTraits<VectorType>::ValueType
VectorReduceImpl(const VectorType &vector,
                 FunctorType functor,
                 dax::VectorTraitsTagMultipleComponents)
{
  typedef dax::VectorTraits<VectorType> Traits;
  typedef typename Traits::ValueType ValueType;
  ValueType result = functor(Traits::GetComponent(vector, 0),
                             Traits::GetComponent(vector, 1));
  for (int component = 2; component < Traits::NUM_COMPONENTS; component++)
    {
    result = functor(result, Traits::GetComponent(vector, component));
    }
  return result;
}

template<class VectorType, class FunctorType>
DAX_EXEC_CONT_EXPORT typename dax::VectorTraits<VectorType>::ValueType
VectorReduceImpl(const VectorType &vector,
                 FunctorType,
                 dax::VectorTraitsTagSingleComponent)
{
  return dax::VectorTraits<VectorType>::GetComponent(vector, 0);
}

}

/// Reduces the components in a vector to a single value.
///
/// \param vector is an input vector of some ambiguous type.
///
/// \param functor is a function or functor class that accepts two arguments of
/// the scalar type contained in the vector (specifically
/// <tt>dax::VectorTraits<VectorType>::ValueType</tt>) and returns a value of
/// the same type.  The operation should be associative.
///
/// \return The reduced value of the vector. If the vector contains only one
/// component, that component is returned.
///
template<class VectorType, class FunctorType>
DAX_EXEC_CONT_EXPORT typename dax::VectorTraits<VectorType>::ValueType
VectorReduce(const VectorType &vector, FunctorType functor)
{
  typedef typename dax::VectorTraits<VectorType>::HasMultipleComponents
      MultipleComponentsTag;
  return dax::internal::VectorReduceImpl(vector,
                                         functor,
                                         MultipleComponentsTag());
}

}

#endif //__dax_VectorOperations_h
