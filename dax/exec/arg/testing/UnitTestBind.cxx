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
#include <dax/exec/arg/FindBinding.h>
#include <dax/exec/internal/WorkletBase.h>

#include <dax/testing/Testing.h>

namespace {

using dax::cont::arg::Field;
using dax::cont::sig::placeholders::_1;

#ifndef FieldIn
# define FieldIn dax::cont::arg::Field(*)(In)
#endif

struct WorkType1 : public dax::exec::internal::WorkletBase

{
  typedef WorkType1 DomainType;
};

struct Worklet1: public WorkType1
{
  typedef void ControlSignature(FieldIn);
  typedef void ExecutionSignature(_1);
};

void Bind()
{
  typedef dax::internal::Invocation<Worklet1,dax::internal::ParameterPack<float> > Invocation1;
  typedef dax::cont::internal::Bindings<Invocation1>::type Bindings1;
  Bindings1 cb1(1.0f,
                dax::internal::MembersInitialArgumentTag(),
                dax::internal::MembersExecContTag());
  dax::exec::arg::FindBinding<Invocation1, _1>::type eb1_1(cb1);
  DAX_TEST_ASSERT(eb1_1(0,Worklet1()) == 1.0f, "Execution environment binding is not 1.0f");
}

}

int UnitTestBind(int, char *[])
{
  return dax::testing::Testing::Run(Bind);
}
