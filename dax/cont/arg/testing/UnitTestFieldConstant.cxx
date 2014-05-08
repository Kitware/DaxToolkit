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
#include <dax/cont/sig/Tag.h>
#include <dax/cont/testing/Testing.h>

namespace{
using dax::cont::arg::Field;

struct WorkType1
{
  typedef WorkType1 WorkType;
};

struct Worklet1: public WorkType1
{
  typedef void ControlSignature(Field);
};

template<typename T>
void verifyConstantExists(T value)
{
  typedef dax::internal::Invocation<Worklet1,dax::internal::ParameterPack<T> > Invocation1;
  typedef typename dax::cont::internal::Bindings<Invocation1>::type Bindings1;
  Bindings1 binded = dax::cont::internal::BindingsCreate(
        Worklet1(), dax::internal::make_ParameterPack(value));
  (void)binded;
}

void FieldConstant()
{
  //confirm that we can bind to the following types:

  //integer
  verifyConstantExists<int>(1);


  //double
  verifyConstantExists<double>(1.35);

  //float
  verifyConstantExists<float>(3.14f);

  //dax tuple
  dax::Tuple<dax::Scalar,6> tuple6;
  tuple6[0]=0.0f; tuple6[1]=0.5f; tuple6[2]=0.25f;
  tuple6[0]=0.0f; tuple6[1]=-0.5f; tuple6[2]=-0.25f;
  verifyConstantExists<dax::Tuple<dax::Scalar,6> >(tuple6);

  //dax::vectors
  dax::Vector2 vec2(-1, -2);
  verifyConstantExists<dax::Vector2>(vec2);

  dax::Vector3 vec3(-1, -2, -3);
  verifyConstantExists<dax::Vector3>(vec3);

  dax::Vector4 vec4(-1, -2, -3, -4);
  verifyConstantExists<dax::Vector4>(vec4);

  //dax::Id3
  dax::Id3 id3(1,2,3);
  verifyConstantExists<dax::Id3>(id3);
}

}

int UnitTestFieldConstant(int, char *[])
{
  return dax::cont::testing::Testing::Run(FieldConstant);
}
