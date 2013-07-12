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
//  Copyright 2013 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __dax_cont_sig_ReductionCount_h
#define __dax_cont_sig_ReductionCount_h

/// \namespace dax::cont::sig::placeholders
/// \brief Placeholders for worklet \c ExecutionSignature declarations.

namespace dax { namespace cont { namespace sig {

/// \headerfile ReductionCount.h dax/cont/sig/ReductionCount.h
/// \brief Reference worklet Id in the \c ExecutionSignature declarations.
class ReductionCount {};

/// \brief Used internally to map an array of reduction counts to a
/// ReductionCount signature.
template<int> class ReductionCountArg {};

/// \headerfile ReductionOffset.h dax/cont/sig/ReductionOffset.h
/// \brief Reference worklet Id in the \c ExecutionSignature declarations.
class ReductionOffset {};

/// \brief Used internally to map an array of reduction counts to a
/// ReductionOffset signature.
template<int> class ReductionOffsetArg {};

/// \headerfile ReductionOffset.h dax/cont/sig/ReductionOffset.h
/// \brief Reference worklet Id in the \c ExecutionSignature declarations.
class ReductionIndexPortal {};

/// \brief Used internally to map an array of reduction counts to a
/// ReductionOffset signature.
template<int> class ReductionIndexPortalArg {};

namespace internal {

/// \brief Converts type T to a Arg placeholder, presumes T::value returns an
/// integer. Is designed as a boost mpl metafunction. Used in, for example, the
/// ExecArgToUseMetaFunc in dax::internal::ReplaceAndExtendSignatures.
///
struct ReductionCountMetaFunc
{
  template<typename T>
  struct apply { typedef dax::cont::sig::ReductionCountArg<T::value> type; };
};

/// \brief Converts type T to a Arg placeholder, presumes T::value returns an
/// integer. Is designed as a boost mpl metafunction. Used in, for example, the
/// ExecArgToUseMetaFunc in dax::internal::ReplaceAndExtendSignatures.
///
struct ReductionOffsetMetaFunc
{
  template<typename T>
  struct apply { typedef dax::cont::sig::ReductionOffsetArg<T::value> type; };
};

/// \brief Converts type T to a Arg placeholder, presumes T::value returns an
/// integer. Is designed as a boost mpl metafunction. Used in, for example, the
/// ExecArgToUseMetaFunc in dax::internal::ReplaceAndExtendSignatures.
///
struct ReductionIndexPortalMetaFunc
{
  template<typename T>
  struct apply { typedef dax::cont::sig::ReductionIndexPortalArg<T::value> type; };
};

} // namespace internal

}}} // namespace dax::cont::sig

#endif //__dax_cont_sig_ReductionCount_h
