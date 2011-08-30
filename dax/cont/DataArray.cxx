/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/DataArray.h>

// FIXME
#include <dax/cont/DataArrayIrregular.h>

// explicit instantiation
template class dax::cont::DataArrayIrregular<dax::Scalar>;
template class dax::cont::DataArrayIrregular<dax::Vector3>;
template class dax::cont::DataArrayIrregular<dax::Vector4>;

//-----------------------------------------------------------------------------
dax::cont::DataArray::DataArray()
{

}

//-----------------------------------------------------------------------------
dax::cont::DataArray::~DataArray()
{
}
