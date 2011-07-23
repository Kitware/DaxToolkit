/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/// Defines various access traits for DaxDataArray
#ifndef __DaxDataArrayTraits_h
#define __DaxDataArrayTraits_h

#include "DaxDataArrayIrregular.h"
#include "DaxDataArrayStructuredConnectivity.h"
#include "DaxDataArrayStructuredPoints.h"

class DaxDataArrayConnectivityTraits
{
public:
  __device__ static DaxId GetNumberOfConnectedElements(
    const DaxWork& work, const DaxDataArray& array)
    {
    switch (array.Type)
      {
    case DaxDataArray::STRUCTURED_CONNECTIVITY:
      return DaxDataArrayStructuredConnectivity::GetNumberOfConnectedElements(
        work, array);
      }
    return -1;
    }

  __device__ static DaxWorkMapField GetConnectedElement(
    const DaxWork& work, const DaxDataArray& connectivityArray,
    DaxId index)
    {
    switch (connectivityArray.Type)
      {
    case DaxDataArray::STRUCTURED_CONNECTIVITY:
      return DaxDataArrayStructuredConnectivity::GetConnectedElement(
        work, connectivityArray, index);
      }
    return DaxWorkMapField();
    }

  __device__ static DaxCellType GetElementsType(
    const DaxDataArray& connectivityArray)
    {
    switch (connectivityArray.Type)
      {
    case DaxDataArray::STRUCTURED_CONNECTIVITY:
      return DaxDataArrayStructuredConnectivity::GetElementsType(
        connectivityArray);
      }

    return EMPTY_CELL;
    }
};

class DaxDataArraySetterTraits
{
public:
  __device__ static void Set(
    const DaxWork& work, DaxDataArray& array, DaxScalar scalar)
    {
    switch (array.Type)
      {
    case DaxDataArray::IRREGULAR:
      return DaxDataArrayIrregular::Set(work, array, scalar);
      }
    }

  __device__ static void Set(
    const DaxWork& work, DaxDataArray& array, DaxVector3 vector3)
    {
    switch (array.Type)
      {
    case DaxDataArray::IRREGULAR:
      return DaxDataArrayIrregular::Set(work, array, vector3);
      }
    }

  __device__ static void Set(
    const DaxWork& work, DaxDataArray& array, DaxVector4 vector4)
    {
    switch (array.Type)
      {
    case DaxDataArray::IRREGULAR:
      return DaxDataArrayIrregular::Set(work, array, vector4);
      }
    }
};

class DaxDataArrayGetterTraits
{
public:
  __device__ static DaxScalar GetScalar(const DaxWork& work, const DaxDataArray& array)
    {
    switch (array.Type)
      {
    case DaxDataArray::IRREGULAR:
      return DaxDataArrayIrregular::GetScalar(work, array);
      }
    return -1;
    }

  __device__ static DaxVector3 GetVector3(const DaxWork& work, const DaxDataArray& array)
    {
    switch (array.Type)
      {
    case DaxDataArray::IRREGULAR:
      return DaxDataArrayIrregular::GetVector3(work, array);

    case DaxDataArray::STRUCTURED_POINTS:
      return DaxDataArrayStructuredPoints::GetVector3(work, array);
      }
    return make_DaxVector3(0, 0, 0);
    }

  __device__ static DaxVector4 GetVector4(const DaxWork& work, const DaxDataArray& array)
    {
    switch (array.Type)
      {
    case DaxDataArray::IRREGULAR:
      return DaxDataArrayIrregular::GetVector4(work, array);
      }
    return make_DaxVector4(0, 0, 0, 0);
    }
};

#endif
