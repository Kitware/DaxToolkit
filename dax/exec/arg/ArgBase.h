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


#ifndef __dax_exec_arg_ArgBase_h
#define __dax_exec_arg_ArgBase_h

#include <dax/Types.h>
#include <dax/exec/internal/WorkletBase.h>

#include <boost/mpl/assert.hpp>
#include <boost/mpl/bool.hpp>
#include <boost/mpl/equal.hpp>
#include <boost/utility/enable_if.hpp>

namespace dax { namespace exec { namespace arg {

//Class that is defined to hold the traits of the derived ArgBase
template<class DerivedArgClass > struct ArgBaseTraits;

template<class DerivedArgClass >
class ArgBase
{
  typedef ArgBaseTraits< DerivedArgClass > DerivedTraits;
  typedef typename DerivedTraits::HasInTag HasInTag;
  typedef typename DerivedTraits::HasOutTag HasOutTag;

  typedef typename DerivedTraits::ReturnType ReturnType;
  typedef typename DerivedTraits::SaveType SaveType;

public:

  template<typename IndexType>
  DAX_EXEC_EXPORT ReturnType operator()(const IndexType& index,
                      const dax::exec::internal::WorkletBase& work)
    {
    return this->readValue(index,work,HasInTag(),HasOutTag());
    }


  DAX_EXEC_EXPORT void SaveExecutionResult(int index,
                        const dax::exec::internal::WorkletBase& work)
    {
    this->saveValue(index,work,HasOutTag());
    }

  DAX_EXEC_EXPORT void SaveExecutionResult(int index,
                        const SaveType& v,
                        const dax::exec::internal::WorkletBase& work)
    {
    this->saveValue(index,v,work,HasOutTag());
    }


private:
  template<typename IndexType>
  DAX_EXEC_EXPORT ReturnType readValue(const IndexType& index,
                       const dax::exec::internal::WorkletBase& work,
                       ::boost::mpl::bool_<true>,
                       ::boost::mpl::bool_<false>) //hasInTag, no out tag
    {
    return static_cast<DerivedArgClass*>(this)->GetValueForReading(index,work);
    }

  template<typename IndexType>
  DAX_EXEC_EXPORT ReturnType readValue(const IndexType&,
                       const dax::exec::internal::WorkletBase&,
                       ::boost::mpl::bool_<false>,
                       ::boost::mpl::bool_<true>) //hasOutTag, no in tag
    {
    return static_cast<DerivedArgClass*>(this)->GetValueForWriting();
    }

  template<typename OutTagIsOn>
  DAX_EXEC_EXPORT void saveValue(dax::Id index,
                  const dax::exec::internal::WorkletBase& work,
                  OutTagIsOn,
                  typename boost::enable_if<typename OutTagIsOn::type>::type* = 0) const
    {
    static_cast<const DerivedArgClass*>(this)->SaveValue(index,work);
    }

  template<typename OutTagIsOn>
  DAX_EXEC_EXPORT void saveValue(dax::Id index,
                  const SaveType &values,
                  const dax::exec::internal::WorkletBase& work,
                  OutTagIsOn,
                  typename boost::enable_if<typename OutTagIsOn::type>::type* = 0) const
    {
    static_cast<const DerivedArgClass*>(this)->SaveValue(index,values,work);
    }


  template<typename OutTagIsOn>
  DAX_EXEC_EXPORT void saveValue(dax::Id,
                  const dax::exec::internal::WorkletBase&,
                  OutTagIsOn,
                  typename boost::disable_if<typename OutTagIsOn::type>::type* = 0) const
    {
    //no need to write out
    }

  template<typename OutTagIsOn>
  DAX_EXEC_EXPORT void saveValue(dax::Id,
                  const SaveType &,
                  const dax::exec::internal::WorkletBase&,
                  OutTagIsOn,
                  typename boost::disable_if<typename OutTagIsOn::type>::type* = 0) const
    {
    //no need to write out
    }
};

} } } //namespace dax::exec::arg


#endif //__dax_exec_arg_ArgBase_h
