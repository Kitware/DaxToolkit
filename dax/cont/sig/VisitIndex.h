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
#ifndef __dax_cont_sig_VisitIndex_h
#define __dax_cont_sig_VisitIndex_h

/// \namespace dax::cont::sig::placeholders
/// \brief Placeholders for worklet \c ExecutionSignature declarations.

namespace dax { namespace cont { namespace sig {

/// \headerfile VisitIndex.h dax/cont/sig/VisitIndex.h
/// \brief Reference worklet Id in the \c ExecutionSignature declarations.
class VisitIndex {};

/// \brief Used internally to map an array of visit indices to a VisitIndex
/// signature.
template<int> class VisitIndexArg {};

namespace internal {

/// \brief Converts type T to a Arg placeholder, presumes T::value returns an
/// integer. Is designed as a boost mpl metafunction. Used in, for example, the
/// ExecArgToUseMetaFunc in dax::internal::ReplaceAndExtendSignatures.
///
struct VisitIndexMetaFunc
{
  template<typename T>
  struct apply { typedef dax::cont::sig::VisitIndexArg<T::value> type; };
};

} // namespace internal

}}} // namespace dax::cont::sig

#endif //__dax_cont_sig_VisitIndex_h
