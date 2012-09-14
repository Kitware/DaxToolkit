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

#include <dax/cont/internal/Bindings.h>
#include <dax/cont/internal/Testing.h>
#include <dax/cont/arg/Field.h>
#include <dax/cont/arg/FieldConstant.h>

#include <dax/exec/internal/WorkletBase.h>

namespace {

using dax::cont::arg::Field;

struct Worklet1 : public dax::exec::internal::WorkletBase
{
  typedef void ControlSignature(Field);
};

struct Worklet2 : public dax::exec::internal::WorkletBase
{
  typedef void ControlSignature(Field,Field);
};

void Bindings()
{
  dax::cont::internal::Bindings<Worklet1(float)> b1(1.0f);
  dax::cont::internal::Bindings<Worklet2(float,float)> b2(1.0f, 2.0f);
  DAX_TEST_ASSERT((b1.Get<1>().GetExecArg()(0,Worklet1()) == 1.0f), "ExecArg value incorrect!");
  DAX_TEST_ASSERT((b2.Get<1>().GetExecArg()(0,Worklet2()) == 1.0f), "ExecArg value incorrect!");
  DAX_TEST_ASSERT((b2.Get<2>().GetExecArg()(0,Worklet2()) == 2.0f), "ExecArg value incorrect!");
}

} // anonymous namespace

int UnitTestBindings(int, char *[])
{
  return dax::cont::internal::Testing::Run(Bindings);
}
