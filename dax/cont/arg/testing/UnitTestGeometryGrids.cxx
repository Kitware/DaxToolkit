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

#include <dax/cont/arg/GeometryUniformGrid.h>
#include <dax/cont/arg/GeometryUnstructuredGrid.h>

#include <dax/cont/testing/Testing.h>
#include <dax/cont/testing/TestingGridGenerator.h>

#include <dax/cont/internal/Bindings.h>
#include <dax/cont/sig/Tag.h>
#include <dax/exec/WorkletMapCell.h>

namespace{
using dax::cont::arg::Geometry;


struct Worklet1: public dax::exec::WorkletMapCell
{
  typedef void ControlSignature(Geometry(In),Geometry(Out));
};

template<typename T, typename U>
void verifyBindingExists(T t, U u)
{
  typedef dax::internal::Invocation<Worklet1,dax::internal::ParameterPack<T,U> > Invocation1;
  typedef typename dax::cont::internal::Bindings<Invocation1>::type Bindings1;
  Bindings1 binded = dax::cont::internal::BindingsCreate(
        Worklet1(), dax::internal::make_ParameterPack(t, u));
  (void)binded;
}

template<typename T, typename U>
void verifyConstBindingExists(const T& t, const U& u)
{
  typedef dax::internal::Invocation<Worklet1,dax::internal::ParameterPack<T,U> > Invocation1;
  typedef typename dax::cont::internal::Bindings<Invocation1>::type Bindings1;
  Bindings1 binded = dax::cont::internal::BindingsCreate(
        Worklet1(), dax::internal::make_ParameterPack(t, u));
  (void)binded;
}

//-----------------------------------------------------------------------------
struct BindTopoGrids
{
  template<typename GridType>
  void operator()(const GridType&) const
    {
    dax::cont::testing::TestGrid<GridType> grid(4);
    verifyBindingExists<GridType,GridType>( grid.GetRealGrid(), grid.GetRealGrid() );
    verifyConstBindingExists<GridType,GridType>( grid.GetRealGrid(), grid.GetRealGrid() );
    }
};

void TopoGrids()
  {
  dax::cont::testing::GridTesting::TryAllGridTypes(BindTopoGrids());
  }
}

int UnitTestGeometryGrids(int, char *[])
{
  return dax::cont::testing::Testing::Run(TopoGrids);
}
