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
#ifndef __dax_math_Numerical_h
#define __dax_math_Numerical_h

// This header file defines some numerical methods that are useful in
// computational geometry.

#include <dax/Types.h>
#include <dax/math/Matrix.h>
#include <dax/math/Sign.h>

namespace dax {
namespace math {

// The NewtonsMethod function was briefly set to be DAX_EXEC_CONT_EXPORT (like
// the rest of the math functions). However, I changed it back because it uses
// callbacks, and those callbacks would have to also be declared
// DAX_EXEC_CONT_EXPORT, which is a pain (and perhaps impossible) for execution
// environment functions. One could argue that because this method is declared
// DAX_EXEC_CONT_EXPORT it should be moved back under dax/exec/math. I would
// not be totally against that, but it would probably be confusing to split the
// math functions like that (which one is where and why?).

/// Uses Newton's method (a.k.a. Newton-Raphson method) to solve a nonlinear
/// system of equations. This function assumes that the number of variables
/// equals the number of equations. Newton's method operates on an iterative
/// evaluate and search. Evaluations are performed using the functors passed
/// into the NewtonsMethod. The first functor returns the NxN matrix of the
/// Jacobian at a given input point. The second functor returns the N tuple
/// that is the function evaluation at the given input point. The input point
/// that evaluates to the desired output, or the closest point found, is
/// returned.
///
template<int Size, class JacobianFunctor, class FunctionFunctor>
DAX_EXEC_EXPORT
dax::Tuple<dax::Scalar,Size>
NewtonsMethod(JacobianFunctor jacobianEvaluator,
              FunctionFunctor functionEvaluator,
              dax::Tuple<dax::Scalar,Size> desiredFunctionOutput,
              dax::Tuple<dax::Scalar,Size> initialGuess
              = dax::Tuple<dax::Scalar,Size>(dax::Scalar(0)),
              dax::Scalar convergeDifference = 1e-3,
              dax::Id maxIterations = 10)
{
  typedef dax::Tuple<dax::Scalar,Size> VectorType;
  typedef dax::math::Matrix<dax::Scalar,Size,Size> MatrixType;

  VectorType x = initialGuess;

  bool converged = false;
  for (dax::Id iteration = 0;
       !converged && (iteration < maxIterations);
       iteration++)
    {
    // For Newton's method, we solve the linear system
    //
    // Jacobian x deltaX = currentFunctionOutput - desiredFunctionOutput
    //
    // The subtraction on the right side simply makes the target of the solve
    // at zero, which is what Newton's method solves for. The deltaX tells us
    // where to move to to solve for a linear system, which we assume will be
    // closer for our nonlinear system.

    MatrixType jacobian = jacobianEvaluator(x);
    VectorType currentFunctionOutput = functionEvaluator(x);

    bool valid;  // Ignored.
    VectorType deltaX =
        dax::math::SolveLinearSystem(
          jacobian,
          currentFunctionOutput - desiredFunctionOutput,
          valid);

    x = x - deltaX;

    converged = true;
    for (int index = 0; index < Size; index++)
      {
      converged &= (dax::math::Abs(deltaX[index]) < convergeDifference);
      }
    }

  // Not checking whether converged.
  return x;
}

}
} // namespace dax::math

#endif //__dax_math_Numerical_h
