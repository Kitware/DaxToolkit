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
/// \brief Map single float values to \c Field worklet parameters.
template <typename Tags> class ConceptMap<Field(Tags), float>
{
public:
  typedef dax::exec::arg::FieldConstant<dax::Scalar> ExecArg;
  ConceptMap(dax::Scalar x): ExecArg_(static_cast<float>(x)) {}
  ExecArg& GetExecArg() { return this->ExecArg_; }
private:
  ExecArg ExecArg_;
};

/// \headerfile FieldConstant.h dax/cont/arg/FieldConstant.h
/// \brief Map single double values to \c Field worklet parameters.
template <typename Tags> class ConceptMap<Field(Tags), double>
{
public:
  typedef dax::exec::arg::FieldConstant<dax::Scalar> ExecArg;
  ConceptMap(double x): ExecArg_(static_cast<dax::Id>(x)) {}
  ExecArg& GetExecArg() { return this->ExecArg_; }
private:
  ExecArg ExecArg_;
};

/// \headerfile FieldConstant.h dax/cont/arg/FieldConstant.h
/// \brief Map single integer values to \c Field worklet parameters.
template <typename Tags> class ConceptMap<Field(Tags), int>
{
public:
  typedef dax::exec::arg::FieldConstant<dax::Id> ExecArg;
  ConceptMap(int x): ExecArg_(static_cast<dax::Id>(x)) {}
  ExecArg& GetExecArg() { return this->ExecArg_; }
private:
  ExecArg ExecArg_;
};


/// \headerfile FieldConstant.h dax/cont/arg/FieldConstant.h
/// \brief Map single dax::Tuple values to \c Field worklet parameters.
template <typename Tags, typename T, int SIZE>
class ConceptMap<Field(Tags), dax::Tuple<T,SIZE> >
{
private:
  typedef dax::Tuple<T,SIZE> Type;
public:
  typedef dax::exec::arg::FieldConstant<Type> ExecArg;
  ConceptMap(const Type& x): ExecArg_(x) {}
  ExecArg& GetExecArg() { return this->ExecArg_; }
private:
  ExecArg ExecArg_;
};

/// \headerfile FieldConstant.h dax/cont/arg/FieldConstant.h
/// \brief Map single dax::Vector2 values to \c Field worklet parameters.
template <typename Tags> class ConceptMap<Field(Tags), dax::Vector2 >
{
private:
  typedef dax::Vector2 Type;
public:
  typedef dax::exec::arg::FieldConstant<Type> ExecArg;
  ConceptMap(const Type& x): ExecArg_(x) {}
  ExecArg& GetExecArg() { return this->ExecArg_; }
private:
  ExecArg ExecArg_;
};

/// \headerfile FieldConstant.h dax/cont/arg/FieldConstant.h
/// \brief Map single dax::Vector3 values to \c Field worklet parameters.
template <typename Tags> class ConceptMap<Field(Tags), dax::Vector3 >
{
private:
  typedef dax::Vector3 Type;
public:
  typedef dax::exec::arg::FieldConstant<Type> ExecArg;
  ConceptMap(const Type& x): ExecArg_(x) {}
  ExecArg& GetExecArg() { return this->ExecArg_; }
private:
  ExecArg ExecArg_;
};

/// \headerfile FieldConstant.h dax/cont/arg/FieldConstant.h
/// \brief Map single dax::Vector4 values to \c Field worklet parameters.
template <typename Tags> class ConceptMap<Field(Tags), dax::Vector4 >
{
private:
  typedef dax::Vector4 Type;
public:
  typedef dax::exec::arg::FieldConstant<Type> ExecArg;
  ConceptMap(const Type& x): ExecArg_(x) {}
  ExecArg& GetExecArg() { return this->ExecArg_; }
private:
  ExecArg ExecArg_;
};

/// \headerfile FieldConstant.h dax/cont/arg/FieldConstant.h
/// \brief Map single dax::Vector3 values to \c Field worklet parameters.
template <typename Tags> class ConceptMap<Field(Tags), dax::Id3 >
{
private:
  typedef dax::Id3 Type;
public:
  typedef dax::exec::arg::FieldConstant<Type> ExecArg;
  ConceptMap(const Type& x): ExecArg_(x) {}
  ExecArg& GetExecArg() { return this->ExecArg_; }
private:
  ExecArg ExecArg_;
};


}}} // namespace dax::cont::arg

#endif //__dax_cont_arg_FieldConstant_h
