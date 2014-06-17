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
#include <dax/cont/arg/TopologyUniformGrid.h>
#include <dax/cont/internal/Bindings.h>
#include <dax/cont/testing/Testing.h>
#include <dax/cont/dispatcher/CollectCount.h>
#include <dax/cont/sig/Tag.h>
#include <dax/cont/UniformGrid.h>
#include <dax/exec/WorkletMapCell.h>
#include <dax/exec/WorkletMapField.h>
#include <dax/Types.h>
#include <vector>

namespace{

using dax::cont::arg::Field;

struct Worklet1 : public dax::exec::WorkletMapField
{
  typedef void ControlSignature(FieldIn);
};

struct Worklet2 : public dax::exec::WorkletMapField
{
  typedef void ControlSignature(Field,Field);
};

struct Worklet3 : public dax::exec::WorkletMapCell
{
  typedef void ControlSignature(Field,Field);
};

struct Worklet4 : public dax::exec::WorkletMapCell
{
  typedef void ControlSignature(TopologyIn,Field);
};


void CollectCount()
{

  //verify that a single constant value returns a collect count of 1
  {
  typedef dax::internal::Invocation<Worklet1,dax::internal::ParameterPack<int> > Invocation1;
  typedef dax::cont::internal::Bindings<Invocation1>::type Bindings1;
  typedef Worklet1::DomainType DomainType;

  int constantFieldArg=4;
  Bindings1 bindings = dax::cont::internal::BindingsCreate(
        Worklet1(), dax::internal::make_ParameterPack(constantFieldArg));

  // Visit each bound argument to determine the count to be scheduled.
  dax::Id count;
  bindings.ForEachCont(dax::cont::dispatcher::CollectCount<DomainType>(count));
  DAX_TEST_ASSERT((count == 1),
                  "CollectCount must be 1 when we have a constant field arg");
  }

  //verify that a single array returns a count equal to its size
  {
  typedef dax::internal::Invocation<
      Worklet1,
      dax::internal::ParameterPack<
        dax::cont::ArrayHandle<dax::Scalar> > > Invocation1;
  typedef dax::cont::internal::Bindings<Invocation1>::type Bindings1;
  typedef Worklet1::DomainType DomainType;


  const dax::Id size(7);
  std::vector<dax::Scalar> f(size); for(int i=0; i <size; ++i) { f[i] = i;}
  dax::cont::ArrayHandle<dax::Scalar> scalarHandle =
                                            dax::cont::make_ArrayHandle(f);
  Bindings1 bindings = dax::cont::internal::BindingsCreate(
        Worklet1(), dax::internal::make_ParameterPack(scalarHandle));

  // Visit each bound argument to determine the count to be scheduled.
  dax::Id count;
  bindings.ForEachCont(dax::cont::dispatcher::CollectCount<DomainType>(count));
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


  typedef dax::internal::Invocation<
      Worklet2,
      dax::internal::ParameterPack<
        dax::cont::ArrayHandle<dax::Scalar>, int> > Invocation2;
  typedef dax::cont::internal::Bindings<Invocation2>::type Bindings2;
  typedef Worklet2::DomainType DomainType;

  Bindings2 bindings = dax::cont::internal::BindingsCreate(
        Worklet2(), dax::internal::make_ParameterPack(scalarHandle,
                                                      constantFieldArg));

  // Visit each bound argument to determine the count to be scheduled.
  dax::Id count;
  bindings.ForEachCont(dax::cont::dispatcher::CollectCount<DomainType>(count));

  DAX_TEST_ASSERT((count == size),
                  "CollectCount was not the length of the array.");


  typedef dax::internal::Invocation<
      Worklet2,
      dax::internal::ParameterPack<
        int, dax::cont::ArrayHandle<dax::Scalar> > > InvertedInvocation2;
  typedef dax::cont::internal::Bindings<InvertedInvocation2>::type
      InvertedBindings2;
  typedef Worklet2 InvertedTwoSigArg(int,dax::cont::ArrayHandle<dax::Scalar>);
  InvertedBindings2 secondBindings = dax::cont::internal::BindingsCreate(
        Worklet2(), dax::internal::make_ParameterPack(constantFieldArg,
                                                      scalarHandle));

  secondBindings.ForEachCont(
                         dax::cont::dispatcher::CollectCount<DomainType>(count));

  DAX_TEST_ASSERT((count == size),
                  "CollectCount was not the length of the array.");
  }


  //verify that when running as WorkletMapCell and we are given
  //no concepts that match the CellDomain we default to a length of one.
  //While this is not the smartest behavior, I want to document and test
  //that this behavior doesn't regress to a count of undefined length.
  //ToDo: If we don't match the given required domain, we fall back to
  //asking for items that match the AnyDomain.
  {
  int validCollectCount = 1;
  int constantFieldArg=4;
  const dax::Id vectorSize(7);

  std::vector<dax::Scalar> f(vectorSize);
  for(int i=0; i <vectorSize; ++i)
    { f[i] = i;}
  dax::cont::ArrayHandle<dax::Scalar> scalarHandle =
                                              dax::cont::make_ArrayHandle(f);


  typedef dax::internal::Invocation<
      Worklet3,
      dax::internal::ParameterPack<
        dax::cont::ArrayHandle<dax::Scalar>, int> > Invocation3;
  typedef dax::cont::internal::Bindings<Invocation3>::type Bindings3;
  typedef Worklet3::DomainType DomainType;

  Bindings3 bindings = dax::cont::internal::BindingsCreate(
        Worklet3(), dax::internal::make_ParameterPack(scalarHandle,
                                                      constantFieldArg));

  // Visit each bound argument to determine the count to be scheduled.
  dax::Id count;
  bindings.ForEachCont(dax::cont::dispatcher::CollectCount<DomainType>(count));

  //verify that count is equal to one, not the length of the vector, see
  //above comment for why
  DAX_TEST_ASSERT((count == validCollectCount),
                  "CollectCount wasn't 1 which is expected.");


  typedef dax::internal::Invocation<
      Worklet3,
      dax::internal::ParameterPack<
        int, dax::cont::ArrayHandle<dax::Scalar> > > InvertedInvocation3;
  typedef dax::cont::internal::Bindings<InvertedInvocation3>::type
      InvertedBindings3;
  InvertedBindings3 secondBindings = dax::cont::internal::BindingsCreate(
        Worklet3(), dax::internal::make_ParameterPack(constantFieldArg,
                                                      scalarHandle));

  secondBindings.ForEachCont(
                         dax::cont::dispatcher::CollectCount<DomainType>(count));

  //verify that count is equal to one, not the length of the vector, see
  //above comment for why
  DAX_TEST_ASSERT((count == validCollectCount),
                  "CollectCount wasn't 1 which is expected.");
  }

  //verify that when running as WorkletMapCell and we are given
  //concepts that match the CellDomain we default to a length that is
  //equal to the number of cells.
  {
  const dax::Id dimSize = 3;
  dax::cont::UniformGrid<> grid;
  grid.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(dimSize-1,
                                                       dimSize-1,
                                                       dimSize-1));

  const dax::Id validCollectCount = grid.GetNumberOfCells();
  //make it larger than validCollectCount to verify we only use the info
  //from the CellDomain
  const dax::Id vectorSize(validCollectCount*2);

  std::vector<dax::Scalar> f(vectorSize);
  for(int i=0; i <vectorSize; ++i)
    { f[i] = i;}
  dax::cont::ArrayHandle<dax::Scalar> scalarHandle =
                                              dax::cont::make_ArrayHandle(f);


  typedef dax::internal::Invocation<
      Worklet4,
      dax::internal::ParameterPack<
        dax::cont::UniformGrid<>, dax::cont::ArrayHandle<dax::Scalar> > >
      Invocation4;
  typedef dax::cont::internal::Bindings<Invocation4>::type Bindings4;
  typedef Worklet4::DomainType DomainType;

  Bindings4 bindings = dax::cont::internal::BindingsCreate(
        Worklet4(), dax::internal::make_ParameterPack(grid, scalarHandle));

  // Visit each bound argument to determine the count to be scheduled.
  dax::Id count;
  bindings.ForEachCont(dax::cont::dispatcher::CollectCount<DomainType>(count));
  DAX_TEST_ASSERT((count == validCollectCount),
                  "CollectCount wasn't the number of cells in grid.");
  }


}

}

int UnitTestCollectCount(int, char *[])
{
  return dax::cont::testing::Testing::Run(CollectCount);
}
