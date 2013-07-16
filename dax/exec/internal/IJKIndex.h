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
    this->CachedValue = this->IJK[0] + this->Dims[0]*
                        (this->IJK[1] + this->Dims[1]* this->IJK[2]);
    }

  DAX_EXEC_EXPORT
  operator dax::Id(void) const
  { return this->CachedValue + this->IJK[0]; }

  // DAX_EXEC_EXPORT dax::Id GetValue() const
  //   { return this->CachedValue + this->IJK[0]; }

  DAX_EXEC_EXPORT const dax::Id3 GetIJK() const { return this->IJK; }

  DAX_CONT_EXPORT void SetI(dax::Id v) { this->IJK[0]=v; }

  DAX_CONT_EXPORT void SetJ(dax::Id v) { this->IJK[1]=v; this->UpdateCache(); }

  DAX_CONT_EXPORT void SetK(dax::Id v) { this->IJK[2]=v; this->UpdateCache(); }

  DAX_EXEC_EXPORT
  bool operator == (dax::Id v) const
    { return (this->CachedValue + this->IJK[0]) == v;}

  DAX_EXEC_EXPORT
  bool operator != (dax::Id v) const
    { return (this->CachedValue + this->IJK[0]) != v;}

  DAX_EXEC_EXPORT
  bool operator < (dax::Id v) const
    { return (this->CachedValue + this->IJK[0]) < v;}

  DAX_EXEC_EXPORT
  bool operator >= (dax::Id v) const
    { return (this->CachedValue + this->IJK[0]) >= v;}

  DAX_EXEC_EXPORT dax::Id operator * (dax::Id v) const
    { return (this->CachedValue + this->IJK[0]) * v;}


private:

  DAX_EXEC_CONT_EXPORT void UpdateCache()
    {
    this->CachedValue = this->Dims[0]*
                          (this->IJK[1] + this->Dims[1] * this->IJK[2]);
    }


  dax::Id3 IJK, Dims;
  dax::Id CachedValue;
};

} } }

#endif //__dax_exec_internal_IJKIndex_h
