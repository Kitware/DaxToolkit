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
# ifndef __dax__internal__GetNthType_h
# define __dax__internal__GetNthType_h

#include <boost/function_types/components.hpp>
#include <boost/mpl/at.hpp>

namespace dax { namespace internal {

/// \class GetNthType    GetNthType.h dax/internal/GetNthType.h
/// \tparam N            Index of the type to get.
/// \tparam TypeSequence Type sequence from which to extract elements.
///  Currently the only supported type sequence is a function type
///  of the form \code T0(T1,T2,...)
///  \endcode where index \c 0 is the return type and greater indexes
///  are the positional parameter types.
///
/// \brief Lookup elements of a type sequence by position.
template <unsigned int N, typename TypeSequence>
struct GetNthType
{
private:
  typedef typename boost::function_types::components<TypeSequence>::type ParameterTypes;
public:
  /// Type at index \c N of \c TypeSequence
  typedef typename boost::mpl::at_c<ParameterTypes, N>::type type;

};

}} // namespace dax::internal

#endif
