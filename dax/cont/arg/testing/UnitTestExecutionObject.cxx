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

#include <dax/cont/arg/ExecutionObject.h>
#include <dax/cont/internal/Bindings.h>
#include <dax/cont/sig/Tag.h>
#include <dax/cont/internal/testing/Testing.h>

namespace{
using dax::cont::arg::ExecObject;

struct Functor1 : dax::exec::ExecutionObjectBase
{
  dax::Id operator()(dax::Id x){ return x*x;}
};

struct Functor2 : dax::exec::ExecutionObjectBase
{
  dax::Id operator()(dax::Id x, dax::Id y){ return x*y;}
};

struct Functor3 : dax::exec::ExecutionObjectBase
{
  Functor3(dax::Id base):Base(base){}
  dax::Id operator()(dax::Id x){ return x*Base;}
private:
  dax::Id Base;
};

struct WorkType1
{
  typedef WorkType1 WorkType;
};

struct Worklet1: public WorkType1
{
  typedef void ControlSignature(ExecObject);
};

template<typename T>
void verifyConstantExists(T value)
{
  typedef Worklet1 Invocation1(T);
  dax::cont::internal::Bindings<Invocation1> binded(value);
  (void)binded;
}

void FieldConstant()
{
  //confirm that we can bind to the following types:

  //single parameter
  verifyConstantExists<Functor1>(Functor1());


  //double parameter
  verifyConstantExists<Functor2>(Functor2());

  //functor with construtor
  Functor3 three(2);
  verifyConstantExists<Functor3>(three);

}

}

dax::Id UnitTestExecutionObject(dax::Id, char *[])
{
  return dax::cont::internal::Testing::Run(FieldConstant);
}