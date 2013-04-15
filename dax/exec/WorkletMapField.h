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
#ifndef __dax_exec_WorkletMapField_h
#define __dax_exec_WorkletMapField_h

#include <dax/exec/internal/WorkletBase.h>
#include <dax/cont/arg/Field.h>
#include <dax/cont/sig/Tag.h>

namespace dax { namespace exec {

///----------------------------------------------------------------------------
/// Superclass for worklets that map fields without regard to topology or any
/// other connectivity information.
///
class WorkletMapField : public dax::exec::internal::WorkletBase
{
public:
  typedef dax::cont::sig::AnyDomain DomainType;

  DAX_EXEC_CONT_EXPORT WorkletMapField() { }
protected:
  typedef dax::cont::arg::Field Field;

};

}}

#endif //__dax_exec_WorkletMapField_h
