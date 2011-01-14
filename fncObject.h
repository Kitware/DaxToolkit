/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __fncObject_h
#define __fncObject_h

#include "fncSystemIncludes.h"
#include <boost/shared_ptr.hpp>

/// fncObject is the base class for all fncFramework classes.
class fncObject
{
public:
  fncObject();
  virtual ~fncObject();
};

typedef boost::shared_ptr<fncObject> fncObjectPtr;

#endif
