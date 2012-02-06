/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/cont/DeviceAdapterDebug.h>

#include <dax/cont/internal/TestingDeviceAdapter.h>

int UnitTestDeviceAdapterDebug(int, char *[])
{
  return dax::cont::internal::TestingDeviceAdapter
      <dax::cont::DeviceAdapterDebug>::Run();
}
