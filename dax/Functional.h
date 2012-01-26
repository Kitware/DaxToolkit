/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_Functional_h
#define __dax_Functional_h

#include <dax/internal/ExportMacros.h>
#include <dax/Types.h>

namespace dax
{

/// Identity is a Unary Function that represents the identity function: it takes
/// a single argument \c x, and returns \c x.
template<typename T>
struct identity
{
  DAX_EXEC_CONT_EXPORT const T &operator()(const T &x) const {return x;}
};




/// Identity is a Unary Function that takes a single argument \c x, and returns
/// True if it is the identity of the Type \p T.
template<typename T>
struct is_identity
  {
    //should we store a const value of type T that is equal to identity
    //rather than generating one?
    DAX_EXEC_CONT_EXPORT
    bool operator()(const T &x)
    {
      return (x  == dax::identity<T>()(T()));
    }
};

/// Identity is a Unary Function that takes a single argument \c x, and returns
/// True if it isn't the identity of the Type \p T.
template<typename T>
struct not_identity
  {
    //should we store a const value of type T that is equal to identity
    //rather than generating one?
    DAX_EXEC_CONT_EXPORT
    bool operator()(const T &x)
    {
      return (x  != dax::identity<T>()(T()));
    }
};

}


#endif // __dax_Functional_h
