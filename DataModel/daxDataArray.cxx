/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "daxDataArray.h"
#include "daxDataArrayIrregular.h"

// explicit instantiation
template class daxDataArrayIrregular<DaxScalar>;
template class daxDataArrayIrregular<DaxVector3>;
template class daxDataArrayIrregular<DaxVector4>;

//-----------------------------------------------------------------------------
daxDataArray::daxDataArray()
{

}

//-----------------------------------------------------------------------------
daxDataArray::~daxDataArray()
{
}
