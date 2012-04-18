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
#ifndef __dax_exec_math_Compare_h
#define __dax_exec_math_Compare_h

// This header file defines math functions that do comparisons.

#include <dax/Types.h>
#include <dax/TypeTraits.h>
#include <dax/exec/VectorOperations.h>

#ifndef DAX_CUDA
#include <math.h>
#include <stdlib.h>
#endif

namespace dax {
namespace exec {
namespace math {

#ifdef DAX_USE_DOUBLE_PRECISION
#define DAX_SYS_MATH_FUNCTION(func) func
#else //DAX_USE_DOUBLE_PRECISION
#define DAX_SYS_MATH_FUNCTION(func) func ## f
#endif //DAX_USE_DOUBLE_PRECISION


//-----------------------------------------------------------------------------
/// Returns \p x or \p y, whichever is larger.
///
DAX_EXEC_EXPORT dax::Scalar Max(dax::Scalar x, dax::Scalar y)
{
  return DAX_SYS_MATH_FUNCTION(fmax)(x, y);
}
DAX_EXEC_EXPORT dax::vector2 Max(dax::vector2 x, dax::vector2 y)
{
  return dax::make_vector2(Max(x[0], y[0]), Max(x[1], y[1]));
}
DAX_EXEC_EXPORT dax::Vector3 Max(dax::Vector3 x, dax::Vector3 y)
{
  return dax::make_Vector3(Max(x[0], y[0]), Max(x[1], y[1]), Max(x[2], y[2]));
}
DAX_EXEC_EXPORT dax::Vector4 Max(dax::Vector4 x, dax::Vector4 y)
{
  return dax::make_Vector4(Max(x[0], y[0]),
                           Max(x[1], y[1]),
                           Max(x[2], y[2]),
                           Max(x[3], y[3]));
}
DAX_EXEC_EXPORT dax::Id Max(dax::Id x, dax::Id y)
{
  return (x < y) ? y : x;
}
DAX_EXEC_EXPORT dax::Id3 Max(dax::Id3 x, dax::Id3 y)
{
  return dax::make_Id3(Max(x[0], y[0]), Max(x[1], y[1]), Max(x[2], y[2]));
}

//-----------------------------------------------------------------------------
/// Returns \p x or \p y, whichever is smaller.
///
DAX_EXEC_EXPORT dax::Scalar Min(dax::Scalar x, dax::Scalar y)
{
  return DAX_SYS_MATH_FUNCTION(fmin)(x, y);
}
DAX_EXEC_EXPORT dax::Vector2 Min(dax::Vector2 x, dax::Vector2 y)
{
  return dax::make_Vector2(Min(x[0], y[0]), Min(x[1], y[1]));
}
DAX_EXEC_EXPORT dax::Vector3 Min(dax::Vector3 x, dax::Vector3 y)
{
  return dax::make_Vector3(Min(x[0], y[0]), Min(x[1], y[1]), Min(x[2], y[2]));
}
DAX_EXEC_EXPORT dax::Vector4 Min(dax::Vector4 x, dax::Vector4 y)
{
  return dax::make_Vector4(Min(x[0], y[0]),
                           Min(x[1], y[1]),
                           Min(x[2], y[2]),
                           Min(x[3], y[3]));
}
DAX_EXEC_EXPORT dax::Id Min(dax::Id x, dax::Id y)
{
  return (x < y) ? x : y;
}
DAX_EXEC_EXPORT dax::Id3 Min(dax::Id3 x, dax::Id3 y)
{
  return dax::make_Id3(Min(x[0], y[0]), Min(x[1], y[1]), Min(x[2], y[2]));
}

}
}
} // namespace dax::exec::math

#endif //__dax_exec_math_Compare_h
