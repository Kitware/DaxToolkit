/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __dax_core_cont_Object_h
#define __dax_core_cont_Object_h

#include "Core/Control/SystemIncludes.h"
#include "Core/Control/Macros.h"
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

namespace dax { namespace core { namespace cont {

  daxDeclareClass(Object);

  /// daxObject is the base class for all classes in the Dax Control
  /// Environment.
  class Object
    {
  public:
    Object();
    virtual ~Object();
  private:
    daxDisableCopyMacro(Object)
    };

}}}

#endif
