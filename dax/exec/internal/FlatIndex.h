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
#ifndef __dax_exec_internal_FlatIndex_h
#define __dax_exec_internal_FlatIndex_h

#include <dax/Types.h>

namespace dax { namespace exec { namespace internal {

struct FlatIndex
{
  DAX_EXEC_EXPORT
  explicit FlatIndex(dax::Id i):Index(i){}

  DAX_EXEC_EXPORT
  const dax::Id& value() const { return Index; }

  DAX_EXEC_EXPORT
  bool operator == (dax::Id v) const{ return this->Index == v;}

  DAX_EXEC_EXPORT
  bool operator != (dax::Id v) const{ return this->Index != v;}

  DAX_EXEC_EXPORT
  bool operator < (dax::Id v) const{ return this->Index < v;}

  DAX_EXEC_EXPORT
  bool operator >= (dax::Id v) const{ return this->Index >= v;}

  DAX_EXEC_EXPORT
  dax::Id operator* (dax::Id v) const{ return this->Index * v;}


private:
  dax::Id Index;
};

} } }

#endif //__dax_exec_internal_FlatIndex_h
