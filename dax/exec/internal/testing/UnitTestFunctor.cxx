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

#include <dax/cont/arg/FieldConstant.h>
#include <dax/cont/internal/Bindings.h>
#include <dax/cont/sig/Arg.h>
#include <dax/exec/internal/Functor.h>
#include <dax/exec/internal/WorkletBase.h>
#include <dax/testing/Testing.h>

namespace {
using dax::cont::arg::Field;

struct WorkType1 : public dax::exec::internal::WorkletBase
{
  typedef WorkType1 DomainType;
};

struct Worklet1: public WorkType1
{
  static float TestValue;
  typedef void ControlSignature(Field);
  typedef void ExecutionSignature(_1);
  template <typename T>
  void operator()(T v) const
    {
    TestValue = v;
    }
};

float Worklet1::TestValue = 0;

void Functor()
{
  typedef Worklet1 Invocation1(float);
  dax::cont::internal::Bindings<Invocation1> b1(1.0f);
  Worklet1 w1;
  dax::exec::internal::Functor<Invocation1> f1(w1, b1);
  f1(0);
  DAX_TEST_ASSERT(Worklet1::TestValue == 1.0f, "TestValue is not 1.0f");
}

}

int UnitTestFunctor(int, char *[])
{
  return dax::testing::Testing::Run(Functor);
}
