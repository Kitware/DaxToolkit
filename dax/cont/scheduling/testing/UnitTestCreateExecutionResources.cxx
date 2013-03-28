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
#include <dax/cont/internal/testing/Testing.h>
#include <dax/cont/scheduling/CreateExecutionResources.h>
#include <dax/cont/sig/Tag.h>
#include <dax/exec/WorkletMapField.h>
#include <dax/Types.h>
#include <vector>

#include <dax/cont/ArrayPortalFromIterators.h>
#include <dax/exec/arg/FieldPortal.h>


#define VECTOR_LENGTH 64

namespace dax { namespace cont { namespace arg{

template <typename Tags, typename T>
class ConceptMap< Field(Tags), std::vector<T> >
{
  typedef std::vector<T> ArrayType;

public:
  //ignore constant values when finding size of domain
  typedef typename dax::cont::arg::SupportedDomains<dax::cont::sig::AnyDomain>::Tags DomainTags;
  typedef dax::cont::ArrayPortalFromIterators<const T*> PortalType;
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
    DAX_TEST_ASSERT( (VECTOR_LENGTH==size),
                    "Incorrect allocation length passed to std::vector concept map");

    this->Portal = PortalType(&Array[0],&Array[size+1]);
    }

  DAX_CONT_EXPORT dax::Id GetDomainLength(sig::Domain) const
    {
    //determine the proper work count be seing if we are being used
    //as input or output
    return this->Handle.GetNumberOfValues();
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
  typedef void ControlSignature(Field);
};


void ExecResources()
{
  std::vector<dax::Scalar> v(VECTOR_LENGTH);
    for(int i=0; i <VECTOR_LENGTH; ++i) { v[i] = static_cast<dax::Scalar>(i);}

  typedef Worklet1 Sig(std::vector<dax::Scalar>);

  dax::cont::internal::Bindings<Sig> bindings(v);

  // Visit each bound argument to determine the count to be scheduled.
  const dax::Id count(VECTOR_LENGTH);
  bindings.ForEachCont(
             dax::cont::scheduling::CreateExecutionResources(count));


  for (dax::Id i=0; i < VECTOR_LENGTH; ++i)
    {
    DAX_TEST_ASSERT(  (bindings.Get<1>().GetExecArg()(i,Worklet1()) ==
                      static_cast<dax::Scalar>(i)),
                    "ExecArg value incorrect after creating exec resources");
    }

}

}

#undef VECTOR_LENGTH

int UnitTestCreateExecutionResources(int, char *[])
{
  return dax::cont::internal::Testing::Run(ExecResources);
}
