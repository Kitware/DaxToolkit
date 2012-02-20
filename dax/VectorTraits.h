/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_VectorTraits_h
#define __dax_VectorTraits_h

#include <dax/Types.h>

#include <boost/type_traits/remove_const.hpp>

namespace dax {

/// A tag for vectors that are "true" vectors (i.e. have more than one
/// component).
///
struct VectorTraitsTagMultipleComponents { };

/// A tag for vectors that a really just scalars (i.e. have only one component)
///
struct VectorTraitsTagSingleComponent { };

/// The VectorTraits class gives several static members that define how
/// to use a given type as a vector.
///
template<class VectorType>
struct VectorTraits {
  /// Type of the components in the vector.
  ///
  typedef typename VectorType::ValueType ValueType;

  /// Number of components in the vector.
  ///
  static const int NUM_COMPONENTS = VectorType::NUM_COMPONENTS;

  /// A tag specifying whether this vector has multiple components (i.e. is a
  /// "real" vector). This tag can be useful for creating specialized functions
  /// when a vector is really just a scalar.
  ///
  typedef VectorTraitsTagMultipleComponents HasMultipleComponents;
  // Note in implementing the above: this will only work if all vector types
  // really have multiple components. If that is not the case, there needs to
  // be some template specialization magic to correctly choose.

  /// Returns the value in a given component of the vector.
  ///
  DAX_EXEC_CONT_EXPORT static const ValueType &GetComponent(
      const typename boost::remove_const<VectorType>::type &vector,
      int component) {
    return vector[component];
  }
  DAX_EXEC_CONT_EXPORT static ValueType &GetComponent(
      typename boost::remove_const<VectorType>::type &vector,
      int component) {
    return vector[component];
  }

  /// Changes the value in a given component of the vector.
  ///
  DAX_EXEC_CONT_EXPORT static void SetComponent(VectorType &vector,
                                                int component,
                                                ValueType value) {
    vector[component] = value;
  }
};

namespace internal {
/// Used for overriding VectorTraits for basic scalar types.
///
template<typename ScalarType>
struct VectorTraitsBasic {
  typedef ScalarType ValueType;
  static const int NUM_COMPONENTS = 1;
  typedef VectorTraitsTagSingleComponent HasMultipleComponents;

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
///
template<>
struct VectorTraits<dax::Scalar>
    : public dax::internal::VectorTraitsBasic<dax::Scalar>
{
};
template<>
struct VectorTraits<const dax::Scalar>
    : public dax::internal::VectorTraitsBasic<dax::Scalar>
{
};

/// Allows you to treat a dax::Id as if it were a vector.
///
template<>
struct VectorTraits<dax::Id>
    : public dax::internal::VectorTraitsBasic<dax::Id>
{
};
template<>
struct VectorTraits<const dax::Id>
    : public dax::internal::VectorTraitsBasic<dax::Id>
{
};

}

#endif //__dax_VectorTraits_h
