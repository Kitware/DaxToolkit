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
#ifndef __dax_exec_arg_FieldPortal_h
#define __dax_exec_arg_FieldPortal_h

#include <dax/Types.h>
#include <dax/cont/sig/Tag.h>
#include <dax/exec/Assert.h>
#include <dax/exec/internal/WorkletBase.h>
#include <dax/exec/internal/FieldAccess.h>

#include <boost/mpl/if.hpp>
#include <boost/utility/enable_if.hpp>


namespace dax { namespace exec { namespace arg {

namespace detail
{
template <typename T, typename ReferenceType,
          typename  Tags, class Enable = void>
struct ValueStorage
{
  //empty storage for read only portals

  template<typename PortalType>
  DAX_EXEC_EXPORT ReferenceType Get(int index, PortalType const& portal,
                            const dax::exec::internal::WorkletBase& work) const
    {
    return dax::exec::internal::FieldGet(portal,index,work);
    }

  template<typename PortalType>
  DAX_EXEC_EXPORT void Set(int,PortalType const&,
                           const dax::exec::internal::WorkletBase&) const { }

  template<typename PortalType>
  DAX_EXEC_EXPORT void Set(int,PortalType const&,ReferenceType,
                           const dax::exec::internal::WorkletBase&) const { }

};

template <typename T, typename ReferenceType, typename  Tags>
struct ValueStorage<T, ReferenceType, Tags,
    typename boost::enable_if<typename Tags::template Has<
    dax::cont::sig::Out> >::type>
{
  //actually store a value from the protal
  T Value;

  template<typename PortalType>
  DAX_EXEC_EXPORT ReferenceType Get(int index, PortalType const& portal,
                                    const dax::exec::internal::WorkletBase& work)
    {
    this->Value = dax::exec::internal::FieldGet(portal,index,work);
    return this->Value;
    }

  template<typename PortalType>
  DAX_EXEC_EXPORT void Set(int index, PortalType const& portal,
                           const dax::exec::internal::WorkletBase& work) const
    {
    dax::exec::internal::FieldSet(portal,index,this->Value,work);
    }

  template<typename PortalType>
  DAX_EXEC_EXPORT void Set(int index, PortalType const& portal,
                           ReferenceType v,
                           const dax::exec::internal::WorkletBase& work) const
    {
    dax::exec::internal::FieldSet(portal,index,v,work);
    }

};
}

template <typename T, typename Tags, typename PortalExec, typename PortalConstExec>
class FieldPortal
{
  //What we have to do is use mpl::if_ to determine the type for
  //ExecArg
  typedef typename boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Out>,
                                   PortalExec,
                                   PortalConstExec>::type PortalType;
public:
  typedef T ValueType;
  //if we are going with Out tag we create a value storage that holds a copy
  //otherwise we have to pass a copy, since portals don't have to provide a reference
  typedef typename boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Out>,
                                   ValueType&,
                                   ValueType const>::type ReturnType;

  FieldPortal(): Storage(), Portal(){}

  DAX_EXEC_EXPORT ReturnType operator()(dax::Id index,
                      const dax::exec::internal::WorkletBase& work)
    {
    //if we have the In tag we have local store so use that value,
    //otherwise call the portal directly
    return this->Storage.Get(index, this->Portal,work);
    }

  DAX_EXEC_EXPORT ReturnType operator()(dax::Id index,
                  const dax::exec::internal::WorkletBase& work) const
    {
    //if we have the In tag we have local store so use that value,
    //otherwise call the portal directly
    return this->Storage.Get(index, this->Portal,work);
    }

  //After needs to be tagged on out, since you get call .Set
  //on a input portal as that fails
  DAX_EXEC_EXPORT void SaveExecutionResult(int index,
                  const dax::exec::internal::WorkletBase& work) const
    {
    this->Storage.Set(index,this->Portal,work);
    }

  //After needs to be tagged on out, since you get call .Set
  //on a input portal as that fails
  DAX_EXEC_EXPORT void SaveExecutionResult(int index, ReturnType v,
                   const dax::exec::internal::WorkletBase& work) const
    {
    this->Storage.Set(index,this->Portal,v,work);
    }

  void SetPortal(PortalType p) { this->Portal = p; }
private:
  /*
  Todo for basic portals where the underlying storage is the same
  as the value type we want a flag so that we don't store a copy
  int the execPortal, but instead directly return a reference.
  */

  //determine if we need to store a local copy of the value we get from
  //the portal. Remember a portal can return Vector3 that can't be assigned
  //too as they actually might be created on the fly
  detail::ValueStorage< ValueType, ReturnType, Tags > Storage;
  PortalType Portal;
};


} } } //namespace dax::exec::arg



#endif //__dax_exec_arg_FieldPortal_h
