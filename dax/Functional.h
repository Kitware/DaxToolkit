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

#ifndef __dax_Functional_h
#define __dax_Functional_h

#include <dax/internal/ExportMacros.h>
#include <dax/Types.h>

namespace dax
{
/// Predicate that takes a single argument \c x, and returns
/// True if it isn't the identity of the Type \p T.
template<typename T>
struct not_default_constructor
{
  DAX_EXEC_CONT_EXPORT bool operator()(const T &x)
  {
    return (x  != T());
  }
};
}


#endif // __dax_Functional_h
