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
#ifndef __dax_cont_arg_FieldConstant_h
#define __dax_cont_arg_FieldConstant_h

#include <dax/Types.h>
#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/arg/Field.h>
#include <dax/exec/arg/FieldConstant.h>

namespace dax { namespace cont { namespace arg {

/// \headerfile FieldConstant.h dax/cont/arg/FieldConstant.h
/// \brief Map single values to \c Field worklet parameters.
template <typename Tags> class ConceptMap<Field(Tags), float>
{
public:
  typedef dax::exec::arg::FieldConstant<float> ExecArg;
  ConceptMap(float x): ExecArg_(x) {}
  ExecArg& GetExecArg() { return this->ExecArg_; }
private:
  ExecArg ExecArg_;
};

}}} // namespace dax::cont::arg

#endif //__dax_cont_arg_FieldConstant_h
