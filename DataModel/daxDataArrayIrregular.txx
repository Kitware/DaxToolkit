/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxDataArrayIrregular.h"

#include "DaxDataArray.h"
//-----------------------------------------------------------------------------
template <class T>
daxDataArrayIrregular<T>::daxDataArrayIrregular()
{
}

//-----------------------------------------------------------------------------
template <class T>
daxDataArrayIrregular<T>::~daxDataArrayIrregular()
{
  
}

//-----------------------------------------------------------------------------
template <class T>
bool daxDataArrayIrregular<T>::Convert(DaxDataArray* array)
{
  array->Type = DaxDataArray::IRREGULAR;
  array->DataType = DaxDataArray::type(T());
  array->RawData = this->HeavyData.data();
  array->SizeInBytes = sizeof(T) * this->HeavyData.size();
  return true;
}
