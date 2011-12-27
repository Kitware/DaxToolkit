/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_TypeTraits_h
#define __dax_TypeTraits_h

#include <dax/Types.h>

namespace dax {

/// The VectorTraits class gives several static members that define how
/// to use a given type as a vector.
template<class VectorType>
struct VectorTraits {
  typedef typename VectorType::ValueType ValueType;
  static const int NUM_COMPONENTS = VectorType::NUM_COMPONENTS;

  DAX_EXEC_CONT_EXPORT static const ValueType &GetComponent(
      const VectorType &vector,
      int component) {
    return vector[component];
  }
  DAX_EXEC_CONT_EXPORT static ValueType &GetComponent(VectorType &vector,
                                                      int component) {
    return vector[component];
  }

  DAX_EXEC_CONT_EXPORT static void SetComponent(VectorType &vector,
                                                int component,
                                                ValueType value) {
    vector[component] = value;
  }
};

namespace internal {
/// Used for overriding VectorTraits for basic scalar types.
template<typename ScalarType>
struct VectorTraitsBasic {
  typedef ScalarType ValueType;
  static const int NUM_COMPONENTS = 1;

  DAX_EXEC_CONT_EXPORT static const ValueType &GetComponent(
      const ScalarType &vector,
      int) {
    return vector;
  }
  DAX_EXEC_CONT_EXPORT static ValueType &GetComponent(ScalarType &vector, int) {
    return vector;
  }

  DAX_EXEC_CONT_EXPORT static void SetComponent(ScalarType &vector,
                                                int,
                                                ValueType value) {
    vector = value;
  }
};
}

/// Allows you to treat a dax::Scalar as if it were a vector.
template<>
struct VectorTraits<dax::Scalar>
    : public dax::internal::VectorTraitsBasic<dax::Scalar>
{
};

/// Allows you to treat a dax::Id as if it were a vector.
template<>
struct VectorTraits<dax::Id>
    : public dax::internal::VectorTraitsBasic<dax::Id>
{
};

}

#endif //__dax_TypeTraits_h
