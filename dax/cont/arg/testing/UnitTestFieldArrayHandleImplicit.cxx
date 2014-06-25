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

#include <dax/cont/arg/FieldArrayHandleImplicit.h>
#include <dax/cont/testing/Testing.h>

#include <dax/cont/internal/Bindings.h>
#include <dax/cont/sig/Tag.h>
#include <dax/exec/WorkletMapField.h>

namespace{
using dax::cont::arg::Field;

struct Worklet1: public dax::exec::WorkletMapField
{
  typedef void ControlSignature(FieldIn);
};


template<typename T>
void verifyBindingExists(T value)
{
  typedef dax::internal::Invocation<Worklet1,dax::internal::ParameterPack<T> > Invocation1;
  typedef typename dax::cont::internal::Bindings<Invocation1>::type Bindings1;
  Bindings1 binded = dax::cont::internal::BindingsCreate(
        Worklet1(), dax::internal::make_ParameterPack(value));
  (void)binded;
}

template<typename T>
void verifyConstBindingExists(const T value)
{
  typedef dax::internal::Invocation<Worklet1,dax::internal::ParameterPack<T> > Invocation1;
  typedef typename dax::cont::internal::Bindings<Invocation1>::type Bindings1;
  Bindings1 binded = dax::cont::internal::BindingsCreate(
        Worklet1(), dax::internal::make_ParameterPack(value));
  (void)binded;
}

template<typename ValueType>
struct IndexSquared
{
  DAX_EXEC_CONT_EXPORT
  ValueType operator()(dax::Id i) const
    { return ValueType(i*i); }
};


void ArrayHandle()
{
  //confirm that we can bind to the following types with an
  //implicit array handle:

  //integer
  typedef dax::cont::ArrayHandleImplicit<dax::Id,IndexSquared<dax::Id> > IdAType;
  verifyBindingExists<IdAType>( IdAType() );
  verifyConstBindingExists<IdAType>( IdAType() );

  //scalar
  typedef dax::cont::ArrayHandleImplicit<dax::Scalar,IndexSquared<dax::Id> > ScalarAType;
  verifyBindingExists<ScalarAType>( ScalarAType() );
  verifyConstBindingExists<ScalarAType>( ScalarAType() );

  //vector
  typedef dax::cont::ArrayHandleImplicit<dax::Vector2,IndexSquared<dax::Id> > VecAType;
  verifyBindingExists<VecAType>( VecAType() );
  verifyConstBindingExists<VecAType>( VecAType() );

}

}

int UnitTestFieldArrayHandleImplicit(int, char *[])
{
  return dax::cont::testing::Testing::Run(ArrayHandle);
}
