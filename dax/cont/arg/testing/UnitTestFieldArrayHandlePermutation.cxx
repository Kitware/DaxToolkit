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

#include <dax/cont/arg/FieldArrayHandlePermutation.h>
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

void ArrayHandle()
{
  typedef dax::cont::ArrayHandleCounting<dax::Id>  IdCountingArrayHandleType;
  typedef dax::cont::ArrayHandleCounting<dax::Scalar> ScalarCountingArrayHandleType;
  typedef dax::cont::ArrayHandleCounting<dax::Vector2> VectorCountingArrayHandleType;

  IdCountingArrayHandleType id_counting(dax::Id(0),10);
  ScalarCountingArrayHandleType scalar_counting(dax::Scalar(1.0),10);
  VectorCountingArrayHandleType vec_counting(dax::Vector2(1.0),10);

  dax::cont::ArrayHandle<dax::Id> iah;
  dax::cont::ArrayHandle<dax::Scalar> sah;
  dax::cont::ArrayHandle<dax::Vector2> vah;

  //integer + integer
  typedef dax::cont::ArrayHandlePermutation<
      dax::cont::ArrayHandle<dax::Id>, IdCountingArrayHandleType > IdAType;
  verifyBindingExists<IdAType>( IdAType(iah,id_counting) );
  verifyConstBindingExists<IdAType>( IdAType(iah,id_counting) );

  //scalar + counting scalar
  typedef dax::cont::ArrayHandlePermutation<
      dax::cont::ArrayHandle<dax::Scalar>, ScalarCountingArrayHandleType> ScalarAType;
  verifyBindingExists<ScalarAType>( ScalarAType(sah,scalar_counting) );
  verifyConstBindingExists<ScalarAType>( ScalarAType(sah,scalar_counting) );

  //vector + counting scalar
  typedef dax::cont::ArrayHandlePermutation<
      dax::cont::ArrayHandle<dax::Vector2>, VectorCountingArrayHandleType> VecAType;
  verifyBindingExists<VecAType>( VecAType(vah,vec_counting) );
  verifyConstBindingExists<VecAType>( VecAType(vah,vec_counting) );

  //counting integer + counting integer
  typedef dax::cont::ArrayHandlePermutation<
      dax::cont::ArrayHandleCounting<dax::Id>, IdCountingArrayHandleType > CIdAType;
  verifyBindingExists<CIdAType>( CIdAType(id_counting,id_counting) );
  verifyConstBindingExists<CIdAType>( CIdAType(id_counting,id_counting) );

  //counting integer + counting scalar
  typedef dax::cont::ArrayHandlePermutation<
      dax::cont::ArrayHandleCounting<dax::Id>, ScalarCountingArrayHandleType> CScalarAType;
  verifyBindingExists<CScalarAType>( CScalarAType(id_counting,scalar_counting) );
  verifyConstBindingExists<CScalarAType>( CScalarAType(id_counting,scalar_counting) );

  //counting integer + counting vector
  typedef dax::cont::ArrayHandlePermutation<
      dax::cont::ArrayHandleCounting<dax::Id>, VectorCountingArrayHandleType> CVecAType;
  verifyBindingExists<CVecAType>( CVecAType(id_counting,vec_counting) );
  verifyConstBindingExists<CVecAType>( CVecAType(id_counting,vec_counting) );

}

}

int UnitTestFieldArrayHandlePermutation(int, char *[])
{
  return dax::cont::testing::Testing::Run(ArrayHandle);
}
