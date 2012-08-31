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

/// \headerfile Tag.h dax/cont/sig/Tag.h
/// \brief Mark in/out parameters in a worklet \c ControlSignature.
class InOut: public In, public Out {};

}}} // namespace dax::cont::sig

#endif //__dax_cont_sig_Tag_h
