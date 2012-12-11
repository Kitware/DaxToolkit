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
#define DAX_ARRAY_CONTAINER_CONTROL DAX_ARRAY_CONTAINER_CONTROL_BASIC
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_SERIAL

#include <dax/cont/DeviceAdapter.h>

#include <dax/cont/arg/Field.h>
#include <dax/cont/arg/FieldArrayHandle.h>
#include <dax/cont/arg/FieldConstant.h>
#include <dax/cont/internal/Bindings.h>
#include <dax/cont/internal/testing/Testing.h>
#include <dax/cont/scheduling/CollectCount.h>
#include <dax/cont/sig/Tag.h>
#include <dax/exec/WorkletMapField.h>
#include <dax/Types.h>
#include <vector>

namespace{

using dax::cont::arg::Field;

struct Worklet1 : public dax::exec::WorkletMapField
{
  typedef void ControlSignature(Field);
};

struct Worklet2 : public dax::exec::WorkletMapField
{
  typedef void ControlSignature(Field,Field);
};

void CollectCount()
{

  //verify that a single constant value returns a collect count of 1
  {
  typedef Worklet1 Sig(int);
  typedef Worklet1::DomainType DomainType;

  int constantFieldArg=4;
  dax::cont::internal::Bindings<Sig> bindings(constantFieldArg);

  // Visit each bound argument to determine the count to be scheduled.
  dax::Id count;
  bindings.ForEachCont(dax::cont::scheduling::CollectCount<DomainType>(count));
  DAX_TEST_ASSERT((count == 1),
                  "CollectCount must be 1 when we have a constant field arg");
  }

  //verify that a single array returns a count equal to its size
  {
  typedef Worklet1 Sig(dax::cont::ArrayHandle<dax::Scalar> );
  typedef Worklet1::DomainType DomainType;


  const dax::Id size(7);
  std::vector<dax::Scalar> f(size); for(int i=0; i <size; ++i) { f[i] = i;}
  dax::cont::ArrayHandle<dax::Scalar> scalarHandle =
                                            dax::cont::make_ArrayHandle(f);
  dax::cont::internal::Bindings<Sig> bindings(scalarHandle);

  // Visit each bound argument to determine the count to be scheduled.
  dax::Id count;
  bindings.ForEachCont(dax::cont::scheduling::CollectCount<DomainType>(count));
  DAX_TEST_ASSERT((count == size),
              "CollectCount was not the length of the array.");
  }

  //verify that a single array and a constant value arg
  //returns a count equal to its size
  {
  int constantFieldArg=4;
  const dax::Id size(7);
  std::vector<dax::Scalar> f(size); for(int i=0; i <size; ++i) { f[i] = i;}
  dax::cont::ArrayHandle<dax::Scalar> scalarHandle =
                                              dax::cont::make_ArrayHandle(f);


  typedef Worklet2 TwoArgSig(dax::cont::ArrayHandle<dax::Scalar>, int);
  typedef Worklet2::DomainType DomainType;

  dax::cont::internal::Bindings<TwoArgSig> bindings(scalarHandle,
                                                    constantFieldArg);

  // Visit each bound argument to determine the count to be scheduled.
  dax::Id count;
  bindings.ForEachCont(dax::cont::scheduling::CollectCount<DomainType>(count));

  DAX_TEST_ASSERT((count == size),
                  "CollectCount was not the length of the array.");


  typedef Worklet2 InvertedTwoSigArg(int,dax::cont::ArrayHandle<dax::Scalar>);
  dax::cont::internal::Bindings<InvertedTwoSigArg> secondBindings(
                                                          constantFieldArg,
                                                          scalarHandle);

  secondBindings.ForEachCont(
                         dax::cont::scheduling::CollectCount<DomainType>(count));

  DAX_TEST_ASSERT((count == size),
                  "CollectCount was not the length of the array.");
  }



}

}

int UnitTestCollectCount(int, char *[])
{
  return dax::cont::internal::Testing::Run(CollectCount);
}
