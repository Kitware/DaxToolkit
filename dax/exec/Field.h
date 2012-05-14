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
#ifndef __dax_exec_Field_h
#define __dax_exec_Field_h

#include <dax/Types.h>
#include <dax/exec/Assert.h>

namespace dax { namespace exec {

namespace internal {

/// A policy-like class used to determine the input/output and read/write
/// semantics of a Field class.
///
template<typename T, class ExecutionAdapter>
struct FieldAccessPolicyInput
{
  typedef T ValueType;
  typedef typename ExecutionAdapter::template FieldStructures<ValueType>
      ::IteratorConstType IteratorType;
};

/// A policy-like class used to determine the input/output and read/write
/// semantics of a Field class.
///
template<typename T, class ExecutionAdapter>
struct FieldAccessPolicyOutput
{
  typedef T ValueType;
  typedef typename ExecutionAdapter::template FieldStructures<ValueType>
      ::IteratorType IteratorType;
};

/// A tag to determine the association (points, cells, etc.) of a field.
///
struct FieldAssociationBaseTag { };

/// A tag to determine the association (points, cells, etc.) of a field.
///
struct FieldAssociationCellTag : public FieldAssociationBaseTag { };

/// A tag to determine the association (points, cells, etc.) of a field.
///
struct FieldAssociationPointTag : public FieldAssociationBaseTag { };

/// A tag to determine the association (points, cells, etc.) of a field.
///
struct FieldAssociationCoordinatesTag : public FieldAssociationPointTag { };

class FieldAccess;

/// Base class for all Field objects.
///
template<class Access, class Association>
class FieldBase
{
public:
  typedef typename Access::ValueType ValueType;
  typedef typename Access::IteratorType IteratorType;

  DAX_EXEC_EXPORT FieldBase(IteratorType beginIterator)
    : BeginIterator(beginIterator)
  { }

private:
  IteratorType BeginIterator;

  friend class FieldAccess;
};

} // namespace internal

#define DAX_DEFINE_FIELD_MACRO(name, access, association) \
template<typename T, class ExecutionAdapter> \
class name \
    : public internal::FieldBase< \
        internal::FieldAccessPolicy##access<T, ExecutionAdapter>, \
        internal::FieldAssociation##association##Tag> \
{ \
public: \
  typedef internal::FieldBase< \
      internal::FieldAccessPolicy##access<T, ExecutionAdapter>, \
      internal::FieldAssociation##association##Tag> BaseType; \
  typedef typename BaseType::ValueType ValueType; \
  typedef typename BaseType::IteratorType IteratorType; \
 \
  DAX_EXEC_EXPORT name(IteratorType beginIterator = IteratorType()) \
    : BaseType(beginIterator) \
  { \
  } \
}

/// \brief A handle to general input field data.
///
/// Worklets use this object in conjunction with a work object to retrieve the
/// field data that the worklet has access to.  The different types of
/// field classes specify the access and association semantics.
///
DAX_DEFINE_FIELD_MACRO(FieldIn, Input, Base);

/// \brief A handle to general output field data.
///
/// Worklets use this object in conjunction with a work object to retrieve the
/// field data that the worklet has access to.  The different types of
/// field classes specify the access and association semantics.
///
DAX_DEFINE_FIELD_MACRO(FieldOut, Output, Base);

/// \brief A handle to input point field data.
///
/// Worklets use this object in conjunction with a work object to retrieve the
/// field data that the worklet has access to.  The different types of
/// field classes specify the access and association semantics.
///
DAX_DEFINE_FIELD_MACRO(FieldPointIn, Input, Point);

/// \brief A handle to output point field data.
///
/// Worklets use this object in conjunction with a work object to retrieve the
/// field data that the worklet has access to.  The different types of
/// field classes specify the access and association semantics.
///
DAX_DEFINE_FIELD_MACRO(FieldPointOut, Output, Point);

/// \brief A handle to input cell field data.
///
/// Worklets use this object in conjunction with a work object to retrieve the
/// field data that the worklet has access to.  The different types of
/// field classes specify the access and association semantics.
///
DAX_DEFINE_FIELD_MACRO(FieldCellIn, Input, Cell);

/// \brief A handle to output cell field data.
///
/// Worklets use this object in conjunction with a work object to retrieve the
/// field data that the worklet has access to.  The different types of
/// field classes specify the access and association semantics.
///
DAX_DEFINE_FIELD_MACRO(FieldCellOut, Output, Cell);

/// \brief A handle to input coordinates field data (on points).
///
/// Worklets use this object in conjunction with a work object to retrieve the
/// field data that the worklet has access to.  The different types of
/// field classes specify the access and association semantics.
///
template<class ExecutionAdapter>
class FieldCoordinatesIn
    : public internal::FieldBase<
        internal::FieldAccessPolicyInput<dax::Vector3, ExecutionAdapter>,
        internal::FieldAssociationCoordinatesTag>
{
public:
  typedef internal::FieldBase<
      internal::FieldAccessPolicyInput<dax::Vector3, ExecutionAdapter>,
      internal::FieldAssociationCoordinatesTag> BaseType;
  typedef typename BaseType::ValueType ValueType;
  typedef typename BaseType::IteratorType IteratorType;

  DAX_EXEC_EXPORT FieldCoordinatesIn(IteratorType beginIterator=IteratorType())
    : BaseType(beginIterator)
  {
  }
};

/// \brief A handle to output coordinates field data (on points).
///
/// Worklets use this object in conjunction with a work object to retrieve the
/// field data that the worklet has access to.  The different types of
/// field classes specify the access and association semantics.
///
template<class ExecutionAdapter>
class FieldCoordinatesOut
    : public internal::FieldBase<
        internal::FieldAccessPolicyOutput<dax::Vector3, ExecutionAdapter>,
        internal::FieldAssociationCoordinatesTag>
{
public:
  typedef internal::FieldBase<
      internal::FieldAccessPolicyOutput<dax::Vector3, ExecutionAdapter>,
      internal::FieldAssociationCoordinatesTag> BaseType;
  typedef typename BaseType::ValueType ValueType;
  typedef typename BaseType::IteratorType IteratorType;

  DAX_EXEC_EXPORT FieldCoordinatesOut(IteratorType beginIterator=IteratorType())
    : BaseType(beginIterator)
  {
  }
};

#undef DAX_DEFINE_FIELD_MACRO

}} // namespace dax::exec

#endif
