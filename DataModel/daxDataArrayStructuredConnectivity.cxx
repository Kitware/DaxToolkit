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
bool daxDataArrayStructuredConnectivity::Convert(DaxDataArray* array)
{
  if (this->Superclass::Convert(array))
    {
    array->Type = DaxDataArray::STRUCTURED_CONNECTIVITY;
    return true;
    }
  return false;
}
