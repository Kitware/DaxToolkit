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
#include <dax/internal/Tags.h>
#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/arg/Field.h>
#include <dax/cont/sig/Tag.h>
#include <dax/exec/arg/FieldConstant.h>

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_float.hpp>
#include <boost/type_traits/is_integral.hpp>

namespace dax { namespace cont { namespace arg {

/// \headerfile FieldConstant.h dax/cont/arg/FieldConstant.h
/// \brief Map single float values to \c Field worklet parameters.
template <typename Tags, class T>
class ConceptMap<Field(Tags), T, typename boost::enable_if< boost::is_float<T> >::type>
{
public:
  //ignore constant values when finding size of domain
  typedef typename dax::cont::arg::SupportedDomains<dax::cont::sig::Domain>::Tags DomainTags;
  typedef dax::exec::arg::FieldConstant<dax::Scalar> ExecArg;
  explicit ConceptMap(T x): ExecArg_(static_cast<dax::Scalar>(x)) {}
  ExecArg GetExecArg() { return this->ExecArg_; }
  void ToExecution(dax::Id) const {}
private:
  ExecArg ExecArg_;
};

/// \headerfile FieldConstant.h dax/cont/arg/FieldConstant.h
/// \brief Map single integer values to \c Field worklet parameters.
template <typename Tags, class T>
class ConceptMap<Field(Tags), T, typename boost::enable_if< boost::is_integral<T> >::type>
{
public:
  //ignore constant values when finding size of domain
  typedef typename dax::cont::arg::SupportedDomains<dax::cont::sig::Domain>::Tags DomainTags;
  typedef dax::exec::arg::FieldConstant<T> ExecArg;
  explicit ConceptMap(T x): ExecArg_(x) {}
  ExecArg GetExecArg() { return this->ExecArg_; }
  void ToExecution(dax::Id) const {}
private:
  ExecArg ExecArg_;
};

/// \headerfile FieldConstant.h dax/cont/arg/FieldConstant.h
/// \brief Map single dax::Tuple values to \c Field worklet parameters.
template <typename Tags, typename T, int SIZE>
class ConceptMap<Field(Tags), dax::Tuple<T,SIZE> >
{
  typedef dax::Tuple<T,SIZE> Type;
public:
  //ignore constant values when finding size of domain
  typedef typename dax::cont::arg::SupportedDomains<dax::cont::sig::Domain>::Tags DomainTags;
  typedef dax::exec::arg::FieldConstant<Type> ExecArg;
  explicit ConceptMap(const Type& x): ExecArg_(x) {}
  ExecArg GetExecArg() { return this->ExecArg_; }
  void ToExecution(dax::Id) const {}
private:
  ExecArg ExecArg_;
};

/// \headerfile FieldConstant.h dax/cont/arg/FieldConstant.h
/// \brief Map single dax::Vector2 values to \c Field worklet parameters.
template <typename Tags> class ConceptMap<Field(Tags), dax::Vector2 >
{
  typedef dax::Vector2 Type;
public:
  //ignore constant values when finding size of domain
  typedef typename dax::cont::arg::SupportedDomains<dax::cont::sig::Domain>::Tags DomainTags;
  typedef dax::exec::arg::FieldConstant<Type> ExecArg;
  explicit ConceptMap(const Type& x): ExecArg_(x) {}
  ExecArg GetExecArg() { return this->ExecArg_; }
  void ToExecution(dax::Id) const {}
private:
  ExecArg ExecArg_;
};

/// \headerfile FieldConstant.h dax/cont/arg/FieldConstant.h
/// \brief Map single dax::Vector3 values to \c Field worklet parameters.
template <typename Tags> class ConceptMap<Field(Tags), dax::Vector3 >
{
  typedef dax::Vector3 Type;
public:
  //ignore constant values when finding size of domain
  typedef typename dax::cont::arg::SupportedDomains<dax::cont::sig::Domain>::Tags DomainTags;
  typedef dax::exec::arg::FieldConstant<Type> ExecArg;
  explicit ConceptMap(const Type& x): ExecArg_(x) {}
  ExecArg GetExecArg() { return this->ExecArg_; }
  void ToExecution(dax::Id) const {}
private:
  ExecArg ExecArg_;
};

/// \headerfile FieldConstant.h dax/cont/arg/FieldConstant.h
/// \brief Map single dax::Vector4 values to \c Field worklet parameters.
template <typename Tags> class ConceptMap<Field(Tags), dax::Vector4 >
{
  typedef dax::Vector4 Type;
public:
  //ignore constant values when finding size of domain
  typedef typename dax::cont::arg::SupportedDomains<dax::cont::sig::Domain>::Tags DomainTags;
  typedef dax::exec::arg::FieldConstant<Type> ExecArg;
  explicit ConceptMap(const Type& x): ExecArg_(x) {}
  ExecArg GetExecArg() { return this->ExecArg_; }
  void ToExecution(dax::Id) const {}
private:
  ExecArg ExecArg_;
};

/// \headerfile FieldConstant.h dax/cont/arg/FieldConstant.h
/// \brief Map single dax::Vector3 values to \c Field worklet parameters.
template <typename Tags> class ConceptMap<Field(Tags), dax::Id3 >
{
  typedef dax::Id3 Type;
public:
  //ignore constant values when finding size of domain
  typedef typename dax::cont::arg::SupportedDomains<dax::cont::sig::Domain>::Tags DomainTags;
  typedef dax::exec::arg::FieldConstant<Type> ExecArg;
  explicit ConceptMap(const Type& x): ExecArg_(x) {}
  ExecArg GetExecArg() { return this->ExecArg_; }
  void ToExecution(dax::Id) const {}
private:
  ExecArg ExecArg_;
};


}}} // namespace dax::cont::arg

#endif //__dax_cont_arg_FieldConstant_h
