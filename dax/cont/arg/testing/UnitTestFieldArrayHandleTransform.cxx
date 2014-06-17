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

#include <dax/cont/arg/FieldArrayHandleTransform.h>
#include <dax/cont/testing/Testing.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ArrayHandleCounting.h>
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
struct MySquare
{
  template<typename U>
  DAX_EXEC_EXPORT
  ValueType operator()(U u) const
    { return ValueType(u*u); }
};

template<typename T>
void Verify(T )
{
  typedef MySquare<T> FunctorType;

  //verify Transform work with an underlying concrete array handle
  typedef dax::cont::ArrayHandleTransform< T,
                                           dax::cont::ArrayHandle< dax::Id >,
                                           FunctorType > TransformHandle;

  dax::cont::ArrayHandle< dax::Id > input;
  TransformHandle thandle(input,FunctorType());

  verifyBindingExists<TransformHandle>( thandle );
  verifyConstBindingExists<TransformHandle>( thandle );

  //verify Transform work with an underlying counting array handle
  typedef dax::cont::ArrayHandleTransform< T,
                                    dax::cont::ArrayHandleCounting< dax::Id >,
                                    FunctorType > CountingTransformHandle;

  CountingTransformHandle countingTransformed =
      dax::cont::make_ArrayHandleTransform<T>(
        dax::cont::make_ArrayHandleCounting(dax::Id(0),10),
        FunctorType());

  verifyBindingExists<CountingTransformHandle>( countingTransformed );
  verifyConstBindingExists<CountingTransformHandle>( countingTransformed );
}

void ArrayHandle()
{
  //confirm that we can bind to the following types with an
  //transform array handle:
  Verify(dax::Id());
  Verify(dax::Scalar());
  Verify(dax::Vector2());

}

}

int UnitTestFieldArrayHandleTransform(int, char *[])
{
  return dax::cont::testing::Testing::Run(ArrayHandle);
}
