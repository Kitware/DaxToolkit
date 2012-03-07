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
#ifndef __dax_exec_internal_FieldBuild_h
#define __dax_exec_internal_FieldBuild_h

#include <dax/exec/Field.h>

#include <dax/internal/DataArray.h>
#include <dax/internal/GridTopologys.h>

namespace dax { namespace exec { namespace internal {

DAX_EXEC_CONT_EXPORT dax::exec::FieldCoordinates fieldCoordinatesBuild(
    const dax::internal::TopologyUniform &)
{
  dax::internal::DataArray<dax::Vector3> dummyArray;
  dax::exec::FieldCoordinates field(dummyArray);
  return field;
}

}}}

#endif //__dax_exec_internal_FieldBuild_h
