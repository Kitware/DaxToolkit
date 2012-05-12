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
struct FieldAccessor;
}

/// \brief A handle to field data.
///
/// Worklets use this object in conjunction with a work object to retrieve the
/// field data that the worklet has access to.
///
template<typename T, class ExecutionAdapter>
class Field
{
public:
  typedef T ValueType;
  typedef typename ExecutionAdapter::template FieldStructures<ValueType>
      ::IteratorType IteratorType;
  typedef typename ExecutionAdapter::template FieldStructures<ValueType>
      ::IteratorConstType IteratorConstType;

  DAX_EXEC_EXPORT Field(IteratorType beginIterator = IteratorType())
    : BeginIterator(beginIterator)
  {
  }

private:
  IteratorType BeginIterator;

  friend class internal::FieldAccessor;
};

namespace internal {
struct FieldAccessor
{
private:
  template <typename T>
  DAX_EXEC_EXPORT static bool NotDefaultConstructor(T &x) { return (x != T()); }

public:
  // Note that in these methods the pass-by-reference of the field class serves
  // the important function of causing a compile error if trying to pass a
  // field of incompatable const-ness.

  template<typename T, class ExecutionAdapter, class WorkType>
  DAX_EXEC_EXPORT static
  typename dax::exec::Field<T, ExecutionAdapter>::IteratorType
  GetBeginIterator(
      dax::exec::Field<T, ExecutionAdapter> &field,
      WorkType work)
  {
    DAX_ASSERT_EXEC(NotDefaultConstructor(field.BeginIterator), work);
    return field.BeginIterator;
  }

  template<typename T, class ExecutionAdapter, class WorkType>
  DAX_EXEC_EXPORT static
  typename dax::exec::Field<T, ExecutionAdapter>::IteratorConstType
  GetBeginIterator(const dax::exec::Field<T, ExecutionAdapter> &field,
                   WorkType work)
  {
    DAX_ASSERT_EXEC(NotDefaultConstructor(field.BeginIterator), work);
    return field.BeginIterator;
  }
};
} // namespace internal

/// \brief  A handle to field data that is specifically mapped to points.
///
template<typename T, class ExecutionAdapter>
class FieldPoint : public dax::exec::Field<T, ExecutionAdapter>
{
public:
  typedef T ValueType;
  typedef dax::exec::Field<ValueType, ExecutionAdapter> Superclass;
  typedef typename Superclass::IteratorType IteratorType;
  typedef typename Superclass::IteratorConstType IteratorConstType;

  DAX_EXEC_EXPORT FieldPoint(IteratorType beginIterator)
    : Superclass(beginIterator)
  { }
};

/// \brief A handle to field data that represents the coordinates of vertices.
///
template<class ExecutionAdapter>
class FieldCoordinates
    : public dax::exec::FieldPoint<dax::Vector3, ExecutionAdapter>
{
public:
  typedef dax::Vector3 ValueType;
  typedef dax::exec::FieldPoint<ValueType, ExecutionAdapter> Superclass;
  typedef typename Superclass::IteratorType IteratorType;
  typedef typename Superclass::IteratorConstType IteratorConstType;

  DAX_EXEC_EXPORT FieldCoordinates(IteratorType beginIterator = IteratorType())
    : Superclass(beginIterator)
   { }
};

/// \brief A handle to field dta that is specifically mapped to cells.
///
template<class T, class ExecutionAdapter>
class FieldCell : public dax::exec::Field<T, ExecutionAdapter>
{
public:
  typedef T ValueType;
  typedef dax::exec::Field<ValueType, ExecutionAdapter> Superclass;
  typedef typename Superclass::IteratorType IteratorType;
  typedef typename Superclass::IteratorConstType IteratorConstType;

  DAX_EXEC_EXPORT FieldCell(IteratorType beginIterator)
    : Superclass(beginIterator)
  { }
};

}}

#endif
