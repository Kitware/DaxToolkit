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
#ifndef __dax_cont_arg_FieldArrayHandle_h
#define __dax_cont_arg_FieldArrayHandle_h

#include <dax/Types.h>
#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/arg/Field.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/sig/Tag.h>
#include <dax/exec/arg/FieldPortal.h>
#include <dax/internal/Tags.h>

namespace dax { namespace cont { namespace arg {

/// \headerfile FieldConstant.h dax/cont/arg/FieldConstant.h
/// \brief Map single float values to \c Field worklet parameters.
template <typename Tags, typename T, typename ContainerTag, typename Device>
struct ConceptMap< Field(Tags), dax::cont::ArrayHandle<T, ContainerTag, Device> >
{
private:
  typedef dax::cont::ArrayHandle<T,ContainerTag, Device > HandleType;
public:
  //ignore constant values when finding size of domain
  typedef typename dax::cont::arg::SupportedDomains<dax::cont::sig::AnyDomain>::Tags DomainTags;
  typedef dax::exec::arg::FieldPortal<T,Tags,
          typename HandleType::PortalExecution,
          typename HandleType::PortalConstExecution> ExecArg;
  HandleType Handle;


  ConceptMap(HandleType handle):
    Handle(handle),
    ExecArg_()
    {}

  ExecArg& GetExecArg() { return this->ExecArg_; }

  void ToExecution(dax::Id size, boost::false_type, boost::true_type)
    { /* Output */
    this->ExecArg_.Portal = this->Handle.PrepareForOutput(size);
    }

  void ToExecution(dax::Id, boost::true_type,  boost::false_type)
    { /* Input  */
    this->ExecArg_.Portal = this->Handle.PrepareForInput();
    }

  void ToExecution(dax::Id, boost::true_type,  boost::true_type)
    { /* Input/Output */
    this->ExecArg_.Portal = this->Handle.PrepareForInPlace();
    }

  //we need to pass the number of elements to allocate
  void ToExecution(dax::Id size)
    {
    ToExecution(size,typename Tags::template Has<dax::cont::sig::In>(),
           typename Tags::template Has<dax::cont::sig::Out>());
    }

  dax::Id GetDomainLength(boost::true_type) const
    {
    //if we are being used as an input array we expect to have a valid
    //number of values
    return this->Handle.GetNumberOfValues();
    }

  dax::Id GetDomainLength(boost::false_type) const
    {
    //if we are being used as output we might not be allocated so something
    //else must provide the work count
    return -1;
    }

  dax::Id GetDomainLength(sig::Domain) const
    {
    //determine the proper work count be seing if we are being used
    //as input or output
    return GetDomainLength(typename Tags::template Has<dax::cont::sig::In>() );
    }

private:
  ExecArg ExecArg_;
};

} } } //namespace dax::cont::arg

#endif //__dax_cont_arg_FieldConstant_h
