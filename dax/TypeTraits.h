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
#ifndef __dax__TypeTraits_h
#define __dax__TypeTraits_h

#include <dax/Types.h>

namespace dax {

/// Tag used to identify types that store real (floating-point) numbers. A
/// TypeTraits class will typedef this class to NumericTag if it stores real
/// numbers (or vectors of real numbers).
///
struct TypeTraitsRealTag {};

/// Tag used to identify types that store integer numbers. A TypeTraits class
/// will typedef this class to NumericTag if it stores integer numbers (or
/// vectors of integers).
///
struct TypeTraitsIntegerTag {};

/// Tag used to identify 0 dimensional types (scalars). Scalars can also be
/// treated like vectors when used with VectorTraits. A TypeTraits class will
/// typedef this class to DimensionalityTag.
///
struct TypeTraitsScalarTag {};

/// Tag used to identify 1 dimensional types (vectors). A TypeTraits class will
/// typedef this class to DimensionalityTag.
///
struct TypeTraitsVectorTag {};

template<typename T> struct TypeTraits;

#ifdef DAX_DOXYGEN_ONLY

/// The TypeTraits class provides helpful compile-time information about the
/// basic types used in Dax (and a few others for convienience). The majority
/// of TypeTraits contents are typedefs to tags that can be used to easily
/// override behavior of called functions.
///
template<typename T>
class TypeTraits {
  typedef int tag_type; // Shut up, test compile.
public:

  /// \brief A tag to determing whether the type is integer or real.
  ///
  /// This tag is either TypeTraitsRealTag or TypeTraitsIntegerTag.
  typedef tag_type NumericTag;

  /// \brief A tag to determine whether the type has multiple components.
  ///
  /// This tag is either TypeTraitsScalarTag or TypeTraitsVectorTag. Scalars can
  /// also be treated as vectors.
  typedef tag_type DimensionalityTag;
};

#endif //DAX_DOXYGEN_ONLY

#define DAX_BASIC_REAL_TYPE(T) \
template<> struct TypeTraits<T> { \
  typedef TypeTraitsRealTag NumericTag; \
  typedef TypeTraitsScalarTag DimensionalityTag; \
}

#define DAX_BASIC_INTEGER_TYPE(T) \
template<> struct TypeTraits<T> { \
  typedef TypeTraitsIntegerTag NumericTag; \
  typedef TypeTraitsScalarTag DimensionalityTag; \
}

/// Traits for basic C++ types.
///

DAX_BASIC_REAL_TYPE(float);
DAX_BASIC_REAL_TYPE(double);
DAX_BASIC_INTEGER_TYPE(char);
DAX_BASIC_INTEGER_TYPE(unsigned char);
DAX_BASIC_INTEGER_TYPE(short);
DAX_BASIC_INTEGER_TYPE(unsigned short);
DAX_BASIC_INTEGER_TYPE(int);
DAX_BASIC_INTEGER_TYPE(unsigned int);
#if DAX_SIZE_LONG == 8
DAX_BASIC_INTEGER_TYPE(long);
DAX_BASIC_INTEGER_TYPE(unsigned long);
#elif DAX_SIZE_LONG_LONG == 8
DAX_BASIC_INTEGER_TYPE(long long);
DAX_BASIC_INTEGER_TYPE(unsigned long long);
#else
#error No implementation for 64-bit integer traits.
#endif

#undef DAX_BASIC_REAL_TYPE
#undef DAX_BASIC_INTEGER_TYPE

#define DAX_VECTOR_TYPE(T, NTag) \
template<> struct TypeTraits<T> { \
  typedef NTag NumericTag; \
  typedef TypeTraitsVectorTag DimensionalityTag; \
}

/// Traits for vector types.
///

DAX_VECTOR_TYPE(dax::Id3, TypeTraitsIntegerTag);
DAX_VECTOR_TYPE(dax::Vector2, TypeTraitsRealTag);
DAX_VECTOR_TYPE(dax::Vector3, TypeTraitsRealTag);
DAX_VECTOR_TYPE(dax::Vector4, TypeTraitsRealTag);

#undef DAX_VECTOR_TYPE

/// Traits for tuples.
///
template<typename T, int Size> struct TypeTraits<dax::Tuple<T, Size> > {
  typedef typename TypeTraits<T>::NumericTag NumericTag;
  typedef TypeTraitsVectorTag DimensionalityTag;
};

} // namespace dax

#endif //__dax__TypeTraits_h
