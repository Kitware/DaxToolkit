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
#ifndef __dax_exec_internal_IJKIndex_h
#define __dax_exec_internal_IJKIndex_h

#include <dax/Types.h>

namespace dax { namespace exec { namespace internal {

struct IJKIndex
{
  DAX_CONT_EXPORT
  IJKIndex(dax::Id3 dims):
    IJK(0,0,0),
    Dims(dims)
    {
    this->CachedValue = IJK[0] + Dims[0]*(IJK[1] + Dims[1]*IJK[2]);
    }

  DAX_EXEC_CONT_EXPORT void updateCache()
    {
    this->CachedValue = Dims[0]*(IJK[1] + Dims[1] * IJK[2]);
    }

  DAX_EXEC_EXPORT dax::Id value() const
    { return this->CachedValue + this->IJK[0]; }

  DAX_EXEC_EXPORT dax::Id operator* (dax::Id v) const
    { return (this->CachedValue + this->IJK[0]) * v;}

  DAX_EXEC_EXPORT const dax::Id& i() const { return this->IJK[0]; }

  DAX_EXEC_EXPORT const dax::Id& j() const { return this->IJK[1]; }

  DAX_EXEC_EXPORT const dax::Id& k() const { return this->IJK[2]; }

  DAX_EXEC_EXPORT const dax::Id3 ijk() const { return this->IJK; }

  DAX_CONT_EXPORT void setI(dax::Id v) { this->IJK[0]=v; }

  DAX_CONT_EXPORT void setJ(dax::Id v) { this->IJK[1]=v; this->updateCache(); }

  DAX_CONT_EXPORT void setK(dax::Id v) { this->IJK[2]=v; this->updateCache(); }

private:
  dax::Id3 IJK, Dims;
  dax::Id CachedValue;
};

} } }

#endif //__dax_exec_internal_IJKIndex_h
