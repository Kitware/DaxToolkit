/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxDataArrayStructuredConnectivity.h"

#include "DaxDataArray.h"
//-----------------------------------------------------------------------------
daxDataArrayStructuredConnectivity::daxDataArrayStructuredConnectivity()
{
}

//-----------------------------------------------------------------------------
daxDataArrayStructuredConnectivity::~daxDataArrayStructuredConnectivity()
{
}

//-----------------------------------------------------------------------------
DaxDataArray daxDataArrayStructuredConnectivity::Upload(
  bool copy_heavy_data/*=false*/)
{
  DaxDataArray array = this->Superclass::Upload(copy_heavy_data);
  array.Type = DaxDataArray::STRUCTURED_CONNECTIVITY;
  return array;
}
