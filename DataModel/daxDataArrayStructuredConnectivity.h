/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#ifndef __daxDataArrayStructuredConnectivity_h
#define __daxDataArrayStructuredConnectivity_h

#include "daxDataArrayStructuredPoints.h"

daxDeclareClass(daxDataArrayStructuredConnectivity);

/// daxDataArrayStructuredConnectivity is used for connectivity array for a
/// structured dataset.
class daxDataArrayStructuredConnectivity : public daxDataArrayStructuredPoints
{
public:
   daxDataArrayStructuredConnectivity();
  virtual ~daxDataArrayStructuredConnectivity();
  daxTypeMacro(daxDataArrayStructuredConnectivity, daxDataArrayStructuredPoints);

  virtual DaxDataArray Upload(bool copy_heavy_data=false);

private:
  daxDisableCopyMacro(daxDataArrayStructuredConnectivity);
};

#endif
