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
#include <dax/cont/internal/Bindings.h>
#include <dax/cont/testing/Testing.h>
#include <dax/cont/scheduling/CreateExecutionResources.h>
#include <dax/cont/sig/Tag.h>
#include <dax/exec/WorkletMapField.h>
#include <dax/Types.h>
#include <vector>

#include <dax/cont/internal/ArrayPortalFromIterators.h>
#include <dax/exec/arg/FieldPortal.h>


int EXPECTED_LENGTH;

namespace dax { namespace cont { namespace arg{

template <typename Tags, typename T>
class ConceptMap< Field(Tags), std::vector<T>& >
{
  typedef std::vector<T>& ArrayType;

public:
  //ignore constant values when finding size of domain
  typedef dax::cont::sig::AnyDomain DomainType;
  typedef ::dax::cont::internal::ArrayPortalFromIterators<const T*> PortalType;
  typedef dax::exec::arg::FieldPortal<T,Tags,PortalType> ExecArg;

  ConceptMap(ArrayType array):
    Array(array),
    Portal()
    {}

  DAX_CONT_EXPORT ExecArg GetExecArg()
    {
    return ExecArg(this->Portal);
    }

  //we need to pass the number of elements to allocate
  DAX_CONT_EXPORT void ToExecution(dax::Id size)
    {
    DAX_TEST_ASSERT( (EXPECTED_LENGTH==size),
                    "Incorrect allocation length passed to std::vector concept map");
    if(size > static_cast<dax::Id>(this->Array.size()))
      {
      this->Array.resize(size);
      }
    this->Portal = PortalType(&Array[0],&Array[size+1]);
    }

  DAX_CONT_EXPORT dax::Id GetDomainLength(sig::Domain) const
    {
    //determine the proper work count be seing if we are being used
    //as input or output
    return static_cast<dax::Id>(this->Array.size());
    }

private:
  ArrayType Array;
  PortalType Portal;
};


}}}


namespace{

using dax::cont::arg::Field;

struct Worklet1 : public dax::exec::WorkletMapField
{
  typedef void ControlSignature(Field, Field(Out));
};


void TestCreateExecutionResources(std::size_t size)
{
  std::vector<dax::Scalar> v(VECTOR_LENGTH);
  for(int i=0; i <VECTOR_LENGTH; ++i) { v[i] = static_cast<dax::Scalar>(i);}

  typedef dax::internal::ParameterPack<std::vector<dax::Scalar> >
      Invocation1;
  typedef dax::cont::internal::Bindings<Worklet1, Invocation1>::type Bindings1;

  Bindings1 bindings = dax::cont::internal::BindingsCreate(
        Worklet1(), dax::internal::make_ParameterPack(v));

  // Visit each bound argument to determine the count to be scheduled.
  const dax::Id count(size);
  bindings.ForEachCont(
             dax::cont::scheduling::CreateExecutionResources(count));

  DAX_TEST_ASSERT( out.size() == in.size(),
                  "out vector not allocated to correct size");

  for (std::size_t i=0; i < size; ++i)
    {
    DAX_TEST_ASSERT(  (bindings.Get<1>().GetExecArg()(i,Worklet1()) ==
                      static_cast<dax::Scalar>(i)),
                    "ExecArg value incorrect after creating exec resources");
    }
}

void ExecResources()
{
  //test that we can create output array of size 1
  TestCreateExecutionResources(1);
  TestCreateExecutionResources(64);
}

}

int UnitTestCreateExecutionResources(int, char *[])
{
  return dax::cont::testing::Testing::Run(ExecResources);
}
