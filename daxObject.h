/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __daxObject_h
#define __daxObject_h

#include "daxSystemIncludes.h"
#include "daxMacros.h"
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>

/// daxObject is the base class for all daxFramework classes.
class daxObject
{
public:
  daxObject();
  virtual ~daxObject();
private:
  daxDisableCopyMacro(daxObject)
};

/// declares daxObjectPtr
daxDefinePtrMacro(daxObject)

#endif
