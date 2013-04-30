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
#ifndef __dax_cont_arg_FieldFunctor_h
#define __dax_cont_arg_FieldFunctor_h

#include <dax/Types.h>
#include <dax/internal/Tags.h>
#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/arg/ExecObject.h>
#include <dax/cont/sig/Tag.h>

#include <dax/exec/ExecutionObjectBase.h>
#include <dax/exec/arg/FieldConstant.h>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>

namespace dax { namespace cont { namespace arg {

/// \headerfile FieldFunctor.h dax/cont/arg/FieldFunctor.h
/// \brief Map user defined objects to \c ExecObject worklet parameters.
template< typename Tags, typename UserClass >
class ConceptMap<ExecObject(Tags), UserClass, typename boost::enable_if< boost::is_base_of<dax::exec::ExecutionObjectBase,UserClass> >::type>
{
public:
  //ignore constant values when finding size of domain
  typedef dax::cont::sig::NullDomain DomainTag;
  typedef dax::exec::arg::FieldConstant<UserClass> ExecArg;

  explicit ConceptMap(UserClass f): UserClassInstance(f) {}

  ExecArg GetExecArg() { return this->UserClassInstance; }

  void ToExecution(dax::Id) const {}
private:
  ExecArg UserClassInstance;
};


}}} // namespace dax::cont::arg

#endif //__dax_cont_arg_FieldFunctor_h
