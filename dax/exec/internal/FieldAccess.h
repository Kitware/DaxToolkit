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
#ifndef __dax_exec_internal_FieldAccess_h
#define __dax_exec_internal_FieldAccess_h

#include <dax/Types.h>
#include <dax/exec/Assert.h>

namespace dax { namespace exec { namespace internal {

template<class PortalType, class WorkType>
DAX_EXEC_EXPORT
typename PortalType::ValueType
FieldGet(const PortalType &arrayPortal, dax::Id index, const WorkType &work)
{
  (void)work;  // Shut up compiler.
  DAX_ASSERT_EXEC(index >= 0, work);
  DAX_ASSERT_EXEC(index < arrayPortal.GetNumberOfValues(), work);
  return arrayPortal.Get(index);
}

template<class PortalType, class WorkType>
DAX_EXEC_EXPORT
void FieldSet(const PortalType &arrayPortal,
              dax::Id index,
              typename PortalType::ValueType value,
              const WorkType &work)
{
  (void)work;  // Shut up compiler.
  DAX_ASSERT_EXEC(index >= 0, work);
  DAX_ASSERT_EXEC(index < arrayPortal.GetNumberOfValues(), work);
  arrayPortal.Set(index, value);
}

template<class PortalType, class WorkType, int Size>
DAX_EXEC_EXPORT
void FieldSetMultiple(const PortalType &arrayPortal,
              dax::Id index,
              const dax::Tuple<typename PortalType::ValueType,Size>& values,
              const WorkType &work)
{
  for (int i = 0; i < Size; i++)
    {
    FieldSet(arrayPortal, index+i, values[i], work);
    }
}

template<class PortalType, class WorkType, int Size>
DAX_EXEC_EXPORT
void FieldSetMultiple(const PortalType &arrayPortal,
              dax::Tuple<dax::Id,Size> indices,
              const dax::Tuple<typename PortalType::ValueType,Size>& values,
              const WorkType &work)
{
  for (int i = 0; i < Size; i++)
    {
    FieldSet(arrayPortal, indices[i], values[i], work);
    }
}

template<class PortalType, class WorkType, int Size>
DAX_EXEC_EXPORT
void FieldGetMultiple(const PortalType &arrayPortal,
              dax::Id index,
              dax::Tuple<typename PortalType::ValueType,Size>& values,
              const WorkType &work)
{
  for (int i = 0; i < Size; i++)
    {
    values[i] = FieldGet(arrayPortal, index+i, work);
    }
}

template<class PortalType, class WorkType, int Size>
DAX_EXEC_EXPORT
dax::Tuple<typename PortalType::ValueType, Size>
FieldGetMultiple(const PortalType &arrayPortal,
                 dax::Tuple<dax::Id,Size> indices,
                 const WorkType &work)
{
  dax::Tuple<typename PortalType::ValueType,Size> values;
  for (int i = 0; i < Size; i++)
    {
    values[i] = FieldGet(arrayPortal, indices[i], work);
    }
  return values;
}

}}} // namespace dax::exec::internal

#endif //__dax_exec_internal_FieldAccess_h
