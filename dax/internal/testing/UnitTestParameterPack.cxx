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
//  Copyright 2013 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================

#include <dax/internal/ParameterPack.h>

#include <dax/cont/Timer.h>

#include <dax/cont/testing/Testing.h>

#include <sstream>
#include <string>

namespace {

typedef dax::Id Type1;
const Type1 Arg1 = 1234;

typedef dax::Scalar Type2;
const Type2 Arg2 = 5678.125;

typedef std::string Type3;
const Type3 Arg3("Third argument");

typedef dax::Vector3 Type4;
const Type4 Arg4(1.2, 3.4, 5.6);

typedef dax::Id3 Type5;
const Type5 Arg5(4, 5, 6);

struct ThreeArgFunctor {
  void operator()(const Type1 &a1, const Type2 &a2, const Type3 &a3) const
  {
    std::cout << "In 3 arg functor." << std::endl;

    DAX_TEST_ASSERT(a1 == Arg1, "Arg 1 incorrect.");
    DAX_TEST_ASSERT(a2 == Arg2, "Arg 2 incorrect.");
    DAX_TEST_ASSERT(a3 == Arg3, "Arg 3 incorrect.");
  }
};

struct ThreeArgModifyFunctor {
  void operator()(Type1 &a1, Type2 &a2, Type3 &a3) const
  {
    std::cout << "In 3 arg modify functor." << std::endl;

    a1 = Arg1;
    a2 = Arg2;
    a3 = Arg3;
  }
};

struct GetReferenceFunctor
{
  template<typename T>
  struct ReturnType {
    typedef const typename boost::remove_reference<T>::type *type;
  };

  template<typename T>
  const T *operator()(const T &x) const { return &x; }
};

struct ThreePointerArgFunctor {
  void operator()(const Type1 *a1, const Type2 *a2, const Type3 *a3) const
  {
    std::cout << "In 3 arg functor." << std::endl;

    DAX_TEST_ASSERT(*a1 == Arg1, "Arg 1 incorrect.");
    DAX_TEST_ASSERT(*a2 == Arg2, "Arg 2 incorrect.");
    DAX_TEST_ASSERT(*a3 == Arg3, "Arg 3 incorrect.");
  }
};

struct ThreeArgFunctorWithReturn {
  std::string operator()(const Type1 &a1,
                         const Type2 &a2,
                         const Type3 &a3) const
  {
    std::cout << "In 3 arg functor with return." << std::endl;

    std::stringstream buffer;
    buffer.precision(10);
    buffer << a1 << " " << a2 << " " << a3;
    return buffer.str();
  }
};

struct FiveArgFunctor {
  void operator()(const Type1 &a1,
                  const Type2 &a2,
                  const Type3 &a3,
                  const Type4 &a4,
                  const Type5 &a5) const
  {
    std::cout << "In 5 arg functor." << std::endl;

    DAX_TEST_ASSERT(a1 == Arg1, "Arg 1 incorrect.");
    DAX_TEST_ASSERT(a2 == Arg2, "Arg 2 incorrect.");
    DAX_TEST_ASSERT(a3 == Arg3, "Arg 3 incorrect.");
    DAX_TEST_ASSERT(a4 == Arg4, "Arg 4 incorrect.");
    DAX_TEST_ASSERT(a5 == Arg5, "Arg 5 incorrect.");
  }
};

struct FiveArgSwizzledFunctor {
  void operator()(const Type5 &a5,
                  const Type1 &a1,
                  const Type3 &a3,
                  const Type4 &a4,
                  const Type2 &a2) const
  {
    std::cout << "In 5 arg functor." << std::endl;

    DAX_TEST_ASSERT(a1 == Arg1, "Arg 1 incorrect.");
    DAX_TEST_ASSERT(a2 == Arg2, "Arg 2 incorrect.");
    DAX_TEST_ASSERT(a3 == Arg3, "Arg 3 incorrect.");
    DAX_TEST_ASSERT(a4 == Arg4, "Arg 4 incorrect.");
    DAX_TEST_ASSERT(a5 == Arg5, "Arg 5 incorrect.");
  }
};

struct ThreeFieldObject {
  Type1 Value1;
  Type2 Value2;
  Type3 Value3;

  ThreeFieldObject() {  }

  ThreeFieldObject(const Type1 &a1, const Type2 &a2, const Type3 &a3)
    : Value1(a1), Value2(a2), Value3(a3) {  }

  void CheckValues() const
  {
    std::cout << "Checking constructed arguments." << std::endl;

    DAX_TEST_ASSERT(this->Value1 == Arg1, "Value 1 incorrect.");
    DAX_TEST_ASSERT(this->Value2 == Arg2, "Value 2 incorrect.");
    DAX_TEST_ASSERT(this->Value3 == Arg3, "Value 3 incorrect.");
  }
};

struct LotsOfArgsFunctor {
  void operator()(dax::Scalar arg1,
                  dax::Scalar arg2,
                  dax::Scalar arg3,
                  dax::Scalar arg4,
                  dax::Scalar arg5,
                  dax::Scalar arg6,
                  dax::Scalar arg7,
                  dax::Scalar arg8,
                  dax::Scalar arg9,
                  dax::Scalar arg10) {
    DAX_TEST_ASSERT(arg1 == 1.0, "Got bad argument");
    DAX_TEST_ASSERT(arg2 == 2.0, "Got bad argument");
    DAX_TEST_ASSERT(arg3 == 3.0, "Got bad argument");
    DAX_TEST_ASSERT(arg4 == 4.0, "Got bad argument");
    DAX_TEST_ASSERT(arg5 == 5.0, "Got bad argument");
    DAX_TEST_ASSERT(arg6 == 6.0, "Got bad argument");
    DAX_TEST_ASSERT(arg7 == 7.0, "Got bad argument");
    DAX_TEST_ASSERT(arg8 == 8.0, "Got bad argument");
    DAX_TEST_ASSERT(arg9 == 9.0, "Got bad argument");
    DAX_TEST_ASSERT(arg10 == 10.0, "Got bad argument");

    this->Field +=
      arg1 + arg2 + arg3 + arg4 + arg5 + arg6 + arg7 + arg8 + arg9 + arg10;
  }
  dax::Scalar Field;
};

void TestParameterPack5(
    const dax::internal::ParameterPack<Type1,Type2,Type3,Type4,Type5> params)
{
  std::cout << "Checking 5 parameter pack." << std::endl;
  DAX_TEST_ASSERT(params.GetNumberOfParameters() == 5,
                  "Got wrong number of parameters.");
  DAX_TEST_ASSERT(params.GetArgument<1>() == Arg1, "Arg 1 incorrect.");
  DAX_TEST_ASSERT(params.GetArgument<2>() == Arg2, "Arg 2 incorrect.");
  DAX_TEST_ASSERT(params.GetArgument<3>() == Arg3, "Arg 3 incorrect.");
  DAX_TEST_ASSERT(params.GetArgument<4>() == Arg4, "Arg 4 incorrect.");
  DAX_TEST_ASSERT(params.GetArgument<5>() == Arg5, "Arg 5 incorrect.");

  std::cout << "Checking invocation." << std::endl;
  dax::internal::ParameterPackInvokeCont(FiveArgFunctor(), params);
  dax::internal::ParameterPackInvokeExec(FiveArgFunctor(), params);

  std::cout << "Swizzling parameters with replace." << std::endl;
  params.Replace<1>(Arg5)
      .Replace<2>(Arg1)
      .Replace<5>(Arg2)
      .InvokeCont(FiveArgSwizzledFunctor());
}

void TestParameterPackReference()
{
  std::cout << "Checking having one parameter pack of references to another."
            << std::endl;
  dax::internal::ParameterPack<Type1,Type2,Type3> originalParams =
      dax::internal::make_ParameterPack(Arg1, Arg2, Arg3);

  dax::internal::ParameterPack<Type1 &, Type2 &, Type3 &>
      referenceParams(originalParams,
                      dax::internal::ParameterPackCopyTag(),
                      dax::internal::ParameterPackExecContTag());

  std::cout << "Checking original parameter values." << std::endl;
  DAX_TEST_ASSERT(referenceParams.GetNumberOfParameters() == 3,
                  "Got wrong number of parameters.");
  DAX_TEST_ASSERT(referenceParams.GetArgument<1>() == Arg1, "Arg 1 incorrect.");
  DAX_TEST_ASSERT(referenceParams.GetArgument<2>() == Arg2, "Arg 2 incorrect.");
  DAX_TEST_ASSERT(referenceParams.GetArgument<3>() == Arg3, "Arg 3 incorrect.");

  std::cout << "Modifying reference parameters and checking original."
            << std::endl;
  referenceParams.GetArgument<1>() += 1;
  referenceParams.GetArgument<2>() += 2;
  referenceParams.GetArgument<3>().append(" modified");

  DAX_TEST_ASSERT(originalParams.GetArgument<1>() == Arg1 + 1,
                  "Arg 1 incorrect.");
  DAX_TEST_ASSERT(originalParams.GetArgument<2>() == Arg2 + 2,
                  "Arg 2 incorrect.");
  DAX_TEST_ASSERT(originalParams.GetArgument<3>() == Arg3 + " modified",
                  "Arg 3 incorrect.");
}

void TestParameterPack()
{
  std::cout << "Creating basic parameter pack." << std::endl;
  dax::internal::ParameterPack<Type1,Type2,Type3> params =
      dax::internal::make_ParameterPack(Arg1, Arg2, Arg3);

  std::cout << "Checking parameters." << std::endl;
  DAX_TEST_ASSERT(params.GetNumberOfParameters() == 3,
                  "Got wrong number of parameters.");
  DAX_TEST_ASSERT(params.GetArgument<1>() == Arg1, "Arg 1 incorrect.");
  DAX_TEST_ASSERT(params.GetArgument<2>() == Arg2, "Arg 2 incorrect.");
  DAX_TEST_ASSERT(params.GetArgument<3>() == Arg3, "Arg 3 incorrect.");

  std::cout << "Checking invocation." << std::endl;
  params.InvokeCont(ThreeArgFunctor());
  params.InvokeExec(ThreeArgFunctor());

  std::cout << "Checking invocation with argument modification." << std::endl;
  params.SetArgument<1>(Type1());
  params.SetArgument<2>(Type2());
  params.SetArgument<3>(Type3());
  DAX_TEST_ASSERT(params.GetArgument<1>() != Arg1, "Arg 1 not cleared.");
  DAX_TEST_ASSERT(params.GetArgument<2>() != Arg2, "Arg 2 not cleared.");
  DAX_TEST_ASSERT(params.GetArgument<3>() != Arg3, "Arg 3 not cleared.");

  params.InvokeCont(ThreeArgModifyFunctor());
  DAX_TEST_ASSERT(params.GetArgument<1>() == Arg1, "Arg 1 incorrect.");
  DAX_TEST_ASSERT(params.GetArgument<2>() == Arg2, "Arg 2 incorrect.");
  DAX_TEST_ASSERT(params.GetArgument<3>() == Arg3, "Arg 3 incorrect.");

  std::cout << "Checking transform invocation." << std::endl;
  params.InvokeCont(ThreePointerArgFunctor(), GetReferenceFunctor());
  params.InvokeExec(ThreePointerArgFunctor(), GetReferenceFunctor());

  std::cout << "Checking invocation with return." << std::endl;
  std::string invokeresult =
      dax::internal::ParameterPackInvokeWithReturnCont<std::string>(
        ThreeArgFunctorWithReturn(), params);
  std::cout << "Got result: " << invokeresult << std::endl;
  DAX_TEST_ASSERT(invokeresult == "1234 5678.125 Third argument",
                  "Got bad result from invoke.");
  invokeresult = "";
  invokeresult =
      dax::internal::ParameterPackInvokeWithReturnExec<std::string>(
        ThreeArgFunctorWithReturn(), params);
  std::cout << "Got result: " << invokeresult << std::endl;
  DAX_TEST_ASSERT(invokeresult == "1234 5678.125 Third argument",
                  "Got bad result from invoke.");

  std::cout << "Checking parameters to constructor." << std::endl;
  ThreeFieldObject object =
      dax::internal::ParameterPackConstructCont<ThreeFieldObject>(params);
  object.CheckValues();

  std::cout << "Checking standard 5 arg." << std::endl;
  TestParameterPack5(
        dax::internal::make_ParameterPack(Arg1,Arg2,Arg3,Arg4,Arg5));

  std::cout << "Checking append." << std::endl;
  TestParameterPack5(params
                     .Append(Arg4)
                     .Append(Arg5));

  TestParameterPackReference();

  std::cout << "Checking time to call lots of args lots of times." << std::endl;
  static dax::Id NUM_TRIALS = 50000;
  LotsOfArgsFunctor f;
  dax::cont::Timer<> timer;
  for (dax::Id trial = 0; trial < NUM_TRIALS; trial++)
    {
    f(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f);
    }
  dax::Scalar directCallTime = timer.GetElapsedTime();
  std::cout << "Time for direct call: " << directCallTime << " seconds"
            << std::endl;
  timer.Reset();
  for (dax::Id trial = 0; trial < NUM_TRIALS; trial++)
    {
    dax::internal::make_ParameterPack(
          1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f)
        .InvokeCont(f);
    }
  dax::Scalar packCallTime = timer.GetElapsedTime();
  std::cout << "Time for packed call: " << packCallTime << " seconds"
            << std::endl;
  std::cout << "Pointless result (making sure compiler computes it) "
            << f.Field << std::endl;

#ifdef NDEBUG
  // Do not do this test when optimizations are off.
  DAX_TEST_ASSERT(packCallTime < 1.05*directCallTime,
                  "Packed call time took longer than expected.");
#endif //NDEBUG
}

} // anonymous namespace

int UnitTestParameterPack(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestParameterPack);
}
