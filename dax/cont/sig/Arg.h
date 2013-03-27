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
#ifndef __dax_cont_sig_Arg_h
#define __dax_cont_sig_Arg_h

/// \namespace dax::cont::sig::placeholders
/// \brief Placeholders for worklet \c ExecutionSignature declarations.

namespace dax { namespace cont { namespace sig {

/// \headerfile Arg.h dax/cont/sig/Arg.h
/// \brief Reference worklet \c ControlSignature arguments in \c ExecutionSignature declarations.
template <int> class Arg {};

namespace placeholders {
 typedef Arg<1> _1; ///< Placeholder for \c ControlSignature argument 1.
 typedef Arg<2> _2; ///< Placeholder for \c ControlSignature argument 2.
 typedef Arg<3> _3; ///< Placeholder for \c ControlSignature argument 3.
 typedef Arg<4> _4; ///< Placeholder for \c ControlSignature argument 4.
 typedef Arg<5> _5; ///< Placeholder for \c ControlSignature argument 5.
 typedef Arg<6> _6; ///< Placeholder for \c ControlSignature argument 6.
 typedef Arg<7> _7; ///< Placeholder for \c ControlSignature argument 7.
 typedef Arg<8> _8; ///< Placeholder for \c ControlSignature argument 8.
 typedef Arg<9> _9; ///< Placeholder for \c ControlSignature argument 9.

} // namespace placeholders


/// \headerfile Arg.h dax/cont/sig/Arg.h
/// \brief Converts type T to a Arg placeholder, presumes T::value returns an integer
/// Is designed as a boost mpl metafunction.
struct to_placeholder
{
  template<typename T>
  struct apply { typedef dax::cont::sig::Arg<T::value> type; };
};

}}} // namespace dax::cont::sig

#endif //__dax_cont_sig_Arg_h
