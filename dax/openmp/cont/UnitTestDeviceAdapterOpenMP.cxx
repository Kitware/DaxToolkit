/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include <dax/openmp/cont/DeviceAdapterOpenMP.h>

#include <dax/cont/internal/TestingDeviceAdapter.h>

int UnitTestDeviceAdapterOpenMP(int, char *[])
{
  return dax::cont::internal::TestingDeviceAdapter
      <dax::openmp::cont::DeviceAdapterOpenMP>::Run();
}
