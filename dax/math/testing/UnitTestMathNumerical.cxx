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

#include <dax/math/Numerical.h>

#include <dax/Types.h>
#include <dax/math/Matrix.h>

#include <dax/testing/Testing.h>

namespace {

// We will test Newton's method with the following three functions:
//
// f1(x,y,z) = x^2 + y^2 + z^2
// f2(x,y,z) = 2x - y + z
// f3(x,y,z) = x + y - z
//
// If we want the result of all three equations to be 1, then there are two
// valid solutions: (2/3, -1/3, -2/3) and (2/3, 2/3, 1/3).
struct EvaluateFunctions
{
  dax::Vector3 operator()(dax::Vector3 x) const {
    dax::Vector3 fx;
    fx[0] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2];
    fx[1] = 2*x[0] - x[1] + x[2];
    fx[2] = x[0] + x[1] - x[2];
    return fx;
  }
};
struct EvaluateJacobian
{
  dax::math::Matrix3x3 operator()(dax::Vector3 x) const {
    dax::math::Matrix3x3 jacobian;
    jacobian(0,0) = 2*x[0];  jacobian(0,1) = 2*x[1];  jacobian(0,2) = 2*x[2];
    jacobian(1,0) = 2;       jacobian(1,1) = -1;      jacobian(1,2) = 1;
    jacobian(2,0) = 1;       jacobian(2,1) = 1;       jacobian(2,2) = -1;
    return jacobian;
  }
};

void TestNewtonsMethod()
{
  std::cout << "Testing Newton's Method." << std::endl;

  dax::Vector3 desiredOutput(1, 1, 1);
  dax::Vector3 expected1(2.0/3.0, -1.0/3.0, -2.0/3.0);
  dax::Vector3 expected2(2.0/3.0, 2.0/3.0, 1.0/3.0);

  dax::Vector3 initialGuess;
  for (initialGuess[0] = 0.25; initialGuess[0] <= 1; initialGuess[0] += 0.25)
    {
    for (initialGuess[1] = 0.25; initialGuess[1] <= 1; initialGuess[1] += 0.25)
      {
      for (initialGuess[2] = 0.25; initialGuess[2] <= 1; initialGuess[2] +=0.25)
        {
        std::cout << "   " << initialGuess << std::endl;

        dax::Vector3 solution =
            dax::math::NewtonsMethod(EvaluateJacobian(),
                                     EvaluateFunctions(),
                                     desiredOutput,
                                     initialGuess,
                                     1e-6);

        DAX_TEST_ASSERT(test_equal(solution, expected1)
                        || test_equal(solution, expected2),
                        "Newton's method did not converge to expected result.");
        }
      }
    }
}

void TestNumerical()
{
  TestNewtonsMethod();
}

} // anonymous namespace

int UnitTestMathNumerical(int, char *[])
{
  return dax::testing::Testing::Run(TestNumerical);
}
