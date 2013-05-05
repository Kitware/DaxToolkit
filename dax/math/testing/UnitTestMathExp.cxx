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

#include <dax/math/Exp.h>

#include <dax/Types.h>
#include <dax/exec/VectorOperations.h>

#include <dax/testing/Testing.h>

#include <boost/lambda/lambda.hpp>

//-----------------------------------------------------------------------------
namespace {

const dax::Id NUM_NUMBERS = 5;
const dax::Scalar NumberList[NUM_NUMBERS] = { 0.25, 0.5, 1.0, 2.0, 3.75 };

//-----------------------------------------------------------------------------
void PowTest()
{
  std::cout << "Runing power tests." << std::endl;
  for (dax::Id index = 0; index < NUM_NUMBERS; index++)
    {
    dax::Scalar x = NumberList[index];
    dax::Scalar powx = dax::math::Pow(x, 2.0);
    dax::Scalar sqrx = x*x;
    DAX_TEST_ASSERT(test_equal(powx, sqrx), "Power gave wrong result.");
    }
}

//-----------------------------------------------------------------------------
struct RaiseTo
{
  dax::Scalar Exponent;
  RaiseTo(dax::Scalar exponent) : Exponent(exponent) { }
  dax::Scalar operator()(dax::Scalar base) const {
    return dax::math::Pow(base, this->Exponent);
  }
};

template<class VectorType, class FunctionType>
void RaiseToTest(FunctionType function, dax::Scalar exponent)
{
  for (dax::Id index = 0; index < NUM_NUMBERS; index++)
    {
    VectorType original;
    dax::exec::VectorFill(original, NumberList[index]);

    VectorType mathresult = function(original);

    VectorType raiseresult = dax::exec::VectorMap(original, RaiseTo(exponent));

    DAX_TEST_ASSERT(test_equal(mathresult, raiseresult),
                    "Exponent functions do not agree.");
    }
}

template<class VectorType> struct SqrtFunctor {
  VectorType operator()(VectorType x) const { return dax::math::Sqrt(x); }
};
template<class VectorType>
void SqrtTest()
{
  std::cout << "  Testing Sqrt "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;
  RaiseToTest<VectorType>(SqrtFunctor<VectorType>(), 0.5);
}

template<class VectorType> struct RSqrtFunctor {
  VectorType operator()(VectorType x) const {return dax::math::RSqrt(x);}
};
template<class VectorType>
void RSqrtTest()
{
  std::cout << "  Testing RSqrt "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;
  RaiseToTest<VectorType>(RSqrtFunctor<VectorType>(), -0.5);
}

template<class VectorType> struct CbrtFunctor {
  VectorType operator()(VectorType x) const { return dax::math::Cbrt(x); }
};
template<class VectorType>
void CbrtTest()
{
  std::cout << "  Testing Cbrt "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;
  RaiseToTest<VectorType>(CbrtFunctor<VectorType>(), 1.0/3.0);
}

template<class VectorType> struct RCbrtFunctor {
  VectorType operator()(VectorType x) const {return dax::math::RCbrt(x);}
};
template<class VectorType>
void RCbrtTest()
{
  std::cout << "  Testing RCbrt "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;
  RaiseToTest<VectorType>(RCbrtFunctor<VectorType>(), -1.0/3.0);
}

//-----------------------------------------------------------------------------
struct RaiseBy
{
  dax::Scalar Base;
  dax::Scalar ExponentBias;
  dax::Scalar ResultBias;
  RaiseBy(dax::Scalar base, dax::Scalar exponentbias, dax::Scalar resultbias)
    : Base(base), ExponentBias(exponentbias), ResultBias(resultbias) { }
  dax::Scalar operator()(dax::Scalar exponent) const {
    return dax::math::Pow(this->Base, exponent + this->ExponentBias)
        + this->ResultBias;
  }
};

template<class VectorType, class FunctionType>
void RaiseByTest(FunctionType function,
                 dax::Scalar base,
                 dax::Scalar exponentbias = 0.0,
                 dax::Scalar resultbias = 0.0)
{
  for (dax::Id index = 0; index < NUM_NUMBERS; index++)
    {
    VectorType original;
    dax::exec::VectorFill(original, NumberList[index]);

    VectorType mathresult = function(original);

    VectorType raiseresult
        = dax::exec::VectorMap(original, RaiseBy(base,
                                                 exponentbias,
                                                 resultbias));

    DAX_TEST_ASSERT(test_equal(mathresult, raiseresult),
                    "Exponent functions do not agree.");
    }
}

template<class VectorType> struct ExpFunctor {
  VectorType operator()(VectorType x) const {return dax::math::Exp(x);}
};
template<class VectorType>
void ExpTest()
{
  std::cout << "  Testing Exp "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;
  RaiseByTest<VectorType>(ExpFunctor<VectorType>(), 2.71828183);
}

template<class VectorType> struct Exp2Functor {
  VectorType operator()(VectorType x) const {return dax::math::Exp2(x);}
};
template<class VectorType>
void Exp2Test()
{
  std::cout << "  Testing Exp2 "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;
  RaiseByTest<VectorType>(Exp2Functor<VectorType>(), 2.0);
}

template<class VectorType> struct ExpM1Functor {
  VectorType operator()(VectorType x) const {return dax::math::ExpM1(x);}
};
template<class VectorType>
void ExpM1Test()
{
  std::cout << "  Testing ExpM1 "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;
  RaiseByTest<VectorType>(ExpM1Functor<VectorType>(), 2.71828183, 0.0, -1.0);
}

template<class VectorType> struct Exp10Functor {
  VectorType operator()(VectorType x) const {return dax::math::Exp10(x);}
};
template<class VectorType>
void Exp10Test()
{
  std::cout << "  Testing Exp10 "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;
  RaiseByTest<VectorType>(Exp10Functor<VectorType>(), 10.0);
}

//-----------------------------------------------------------------------------
void Log2Test()
{
  std::cout << "Testing Log2" << std::endl;
  DAX_TEST_ASSERT(test_equal(dax::math::Log2(dax::Scalar(0.25)),
                             dax::Scalar(-2.0)),
                  "Bad value from Log2");
  DAX_TEST_ASSERT(
        test_equal(dax::math::Log2(dax::make_Vector4(0.5, 1.0, 2.0, 4.0)),
                   dax::make_Vector4(-1.0, 0.0, 1.0, 2.0)),
        "Bad value from Log2");
}

template<class VectorType, class FunctionType>
void LogBaseTest(FunctionType function, dax::Scalar base, dax::Scalar bias=0.0)
{
  for (dax::Id index = 0; index < NUM_NUMBERS; index++)
    {
    VectorType original;
    dax::exec::VectorFill(original, NumberList[index]);

    VectorType mathresult = function(original);

    VectorType basevector;
    dax::exec::VectorFill(basevector, base);
    VectorType biased;
    dax::exec::VectorFill(biased, NumberList[index] + bias);

    VectorType logresult
        = dax::math::Log2(biased)/dax::math::Log2(basevector);

    DAX_TEST_ASSERT(test_equal(mathresult, logresult),
                    "Log functions do not agree.");
    }
}

template<class VectorType> struct LogFunctor {
  VectorType operator()(VectorType x) const {return dax::math::Log(x);}
};
template<class VectorType>
void LogTest()
{
  std::cout << "  Testing Log "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;
  LogBaseTest<VectorType>(LogFunctor<VectorType>(), 2.71828183);
}

template<class VectorType> struct Log10Functor {
  VectorType operator()(VectorType x) const {return dax::math::Log10(x);}
};
template<class VectorType>
void Log10Test()
{
  std::cout << "  Testing Log10 "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;
  LogBaseTest<VectorType>(Log10Functor<VectorType>(), 10.0);
}

template<class VectorType> struct Log1PFunctor {
  VectorType operator()(VectorType x) const {return dax::math::Log1P(x);}
};
template<class VectorType>
void Log1PTest()
{
  std::cout << "  Testing Log1P "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;
  LogBaseTest<VectorType>(Log1PFunctor<VectorType>(), 2.71828183, 1.0);
}

//-----------------------------------------------------------------------------
struct TestExpFunctor
{
  template <typename T> void operator()(const T&) const {
    SqrtTest<T>();
    RSqrtTest<T>();
    CbrtTest<T>();
    RCbrtTest<T>();
    ExpTest<T>();
    Exp2Test<T>();
    ExpM1Test<T>();
    Exp10Test<T>();
    LogTest<T>();
    Log10Test<T>();
    Log1PTest<T>();
  }
};

void RunExpTests()
{
  PowTest();
  Log2Test();
  dax::testing::Testing::TryAllTypes(TestExpFunctor(),
                                      dax::testing::Testing::TypeCheckReal());
}

} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestMathExp(int, char *[])
{
  return dax::testing::Testing::Run(RunExpTests);
}
