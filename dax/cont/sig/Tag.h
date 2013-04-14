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
#ifndef __dax_cont_sig_Tag_h
#define __dax_cont_sig_Tag_h

/// \namespace dax::cont::sig
/// \brief Dax Control Environment signature declaration types

namespace dax { namespace cont { namespace sig {

/// \headerfile Tag.h dax/cont/sig/Tag.h
/// \brief Base class for all worklet \c ControlSignature tag types.
///
/// Dax control environment signature tags are collected in a
/// dax::internal::Tags type using this \c Tag as \c TagBase.
class Tag {};

/// \headerfile Tag.h dax/cont/sig/Tag.h
/// \brief Mark input parameters in a worklet \c ControlSignature.
class In: public Tag {};

/// \headerfile Tag.h dax/cont/sig/Tag.h
/// \brief Mark output parameters in a worklet \c ControlSignature.
class Out: public Tag {};

class Domain: public Tag
{
public:
  class NullDomain;
  class AnyDomain;
  class Cell;
  class Point;
};
template <typename T> class DomainTag: public Domain {};
typedef DomainTag<Domain::NullDomain> NullDomain;
typedef DomainTag<Domain::AnyDomain> AnyDomain;
typedef DomainTag<Domain::Cell> Cell;
typedef DomainTag<Domain::Point> Point;

namespace detail {
template <typename T> class PermutedDomain: public Domain {};
} // namespace detail

// The default implementation to permute a cell.
template<typename T> struct MakePermuted
{
  typedef detail::PermutedDomain<T> Type;
};

// Special cases of permuted cells.  I'm not super happy we have to have them.
// Don't have permutations of permutations.
template<typename T> struct MakePermuted<detail::PermutedDomain<T> >
{
  typedef detail::PermutedDomain<T> Type;
};
// Don't permute AnyDomain. Since it can be anything, it doesn't matter of it's
// permuted.
template<> struct MakePermuted<AnyDomain>
{
  typedef AnyDomain Type;
};
// Don't permute NullDomain. Doesn't really matter since it should not match
// anything, but it still makes no sense to permute it.
template<> struct MakePermuted<NullDomain>
{
  typedef NullDomain Type;
};

typedef MakePermuted<Cell>::Type PermutedCell;


}}} // namespace dax::cont::sig

#endif //__dax_cont_sig_Tag_h
