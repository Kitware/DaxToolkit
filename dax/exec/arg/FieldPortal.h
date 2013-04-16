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
#include <dax/VectorTraits.h>

#include <dax/exec/arg/ArgBase.h>
#include <dax/exec/internal/FieldAccess.h>
#include <dax/exec/internal/IJKIndex.h>

#include <boost/mpl/bool.hpp>
#include <boost/mpl/if.hpp>
#include <boost/utility/enable_if.hpp>


namespace dax { namespace exec { namespace arg {

template <typename T, typename Tags, typename PortalType>
class FieldPortal : public dax::exec::arg::ArgBase< FieldPortal<T, Tags, PortalType> >
{
public:
  typedef dax::exec::arg::ArgBaseTraits< FieldPortal< T, Tags, PortalType > > Traits;

  typedef typename Traits::ValueType ValueType;
  typedef typename Traits::ReturnType ReturnType;
  typedef typename Traits::SaveType SaveType;

  // If you do not explicitly initialize Value, then gcc sometimes complains
  // about an uninitialized value. I don't think it is ever actually used
  // uninitialized (at this time), but the optimizer's execution flow analysis
  // seems to think so.
  FieldPortal(const PortalType& portal):
    Value(typename dax::VectorTraits<ValueType>::ComponentType()),
    Portal(portal)
    {
    }

  DAX_EXEC_EXPORT ReturnType GetValueForWriting()
    { return this->Value; }

  DAX_EXEC_EXPORT ReturnType GetValueForReading(
                                const dax::exec::internal::IJKIndex& index,
                                const dax::exec::internal::WorkletBase& work)
    {
    this->Value = dax::exec::internal::FieldGet(this->Portal,index.GetValue(),work);
    return this->Value;
    }

  DAX_EXEC_EXPORT ReturnType GetValueForReading(
                                dax::Id index,
                                const dax::exec::internal::WorkletBase& work)
    {
    this->Value = dax::exec::internal::FieldGet(this->Portal,index,work);
    return this->Value;
    }

  DAX_EXEC_EXPORT void SaveValue(int index,
                        const dax::exec::internal::WorkletBase& work) const
    {
    dax::exec::internal::FieldSet(Portal,index,this->Value,work);
    }

  DAX_EXEC_EXPORT void SaveValue(int index, const SaveType& v,
                        const dax::exec::internal::WorkletBase& work) const
    {
    dax::exec::internal::FieldSet(Portal,index,v,work);
    }

private:
  ValueType Value;
  PortalType Portal;
};

//the traits for FieldPortal
template <typename T, typename Tags, typename PortalType>
struct ArgBaseTraits< dax::exec::arg::FieldPortal< T, Tags, PortalType > >
{
  typedef typename ::boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Out>,
                                   ::boost::true_type,
                                   ::boost::false_type>::type HasOutTag;

  typedef typename ::boost::mpl::if_<typename Tags::template Has<dax::cont::sig::In>,
                                   ::boost::true_type,
                                   ::boost::false_type>::type HasInTag;
  typedef T ValueType;

  typedef typename ::boost::mpl::if_<typename HasOutTag::type,
                                   T&,
                                   T const& >::type ReturnType;
  typedef ValueType SaveType;
};

} } } //namespace dax::exec::arg



#endif //__dax_exec_arg_FieldPortal_h
