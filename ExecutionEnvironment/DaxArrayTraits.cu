/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/// Defines various access traits for DaxArray
#ifndef __DaxArrayTraits_h
#define __DaxArrayTraits_h

#include "DaxArrayIrregular.cu"
#include "DaxArrayStructuredConnectivity.cu"
#include "DaxArrayStructuredPoints.cu"

class DaxArrayConnectivityTraits
{
public:
  __device__ static DaxId GetNumberOfConnectedElements(
    const DaxWork& work, const DaxArray& array)
    {
    switch (array.Type)
      {
    case DaxArray::STRUCTURED_CONNECTIVITY:
      return DaxArrayStructuredConnectivity::GetNumberOfConnectedElements(
        work, array);
      }
    return -1;
    }

  __device__ static DaxWorkMapField GetConnectedElement(
    const DaxWork& work, const DaxArray& connectivityArray,
    DaxId index)
    {
    switch (connectivityArray.Type)
      {
    case DaxArray::STRUCTURED_CONNECTIVITY:
      return DaxArrayStructuredConnectivity::GetConnectedElement(
        work, connectivityArray, index);
      }
    return DaxWorkMapField();
    }

  __device__ static DaxCellType GetElementsType(
    const DaxArray& connectivityArray)
    {
    switch (connectivityArray.Type)
      {
    case DaxArray::STRUCTURED_CONNECTIVITY:
      return DaxArrayStructuredConnectivity::GetElementsType(
        connectivityArray);
      }

    return EMPTY_CELL;
    }
};

class DaxArraySetterTraits
{
public:
  __device__ static void Set(
    const DaxWork& work, DaxArray& array, DaxScalar scalar)
    {
    switch (array.Type)
      {
    case DaxArray::IRREGULAR:
      return DaxArrayIrregular::Set(work, array, scalar);
      }
    }

  __device__ static void Set(
    const DaxWork& work, DaxArray& array, DaxVector3 vector3)
    {
    switch (array.Type)
      {
    case DaxArray::IRREGULAR:
      return DaxArrayIrregular::Set(work, array, vector3);
      }
    }

  __device__ static void Set(
    const DaxWork& work, DaxArray& array, DaxVector4 vector4)
    {
    switch (array.Type)
      {
    case DaxArray::IRREGULAR:
      return DaxArrayIrregular::Set(work, array, vector4);
      }
    }
};

class DaxArrayGetterTraits
{
public:
  __device__ static DaxScalar GetScalar(const DaxWork& work, const DaxArray& array)
    {
    switch (array.Type)
      {
    case DaxArray::IRREGULAR:
      return DaxArrayIrregular::GetScalar(work, array);
      }
    return -1;
    }

  __device__ static DaxVector3 GetVector3(const DaxWork& work, const DaxArray& array)
    {
    switch (array.Type)
      {
    case DaxArray::IRREGULAR:
      return DaxArrayIrregular::GetVector3(work, array);

    case DaxArray::STRUCTURED_POINTS:
      return DaxArrayStructuredPoints::GetVector3(work, array);
      }
    return make_DaxVector3(0, 0, 0);
    }

  __device__ static DaxVector4 GetVector4(const DaxWork& work, const DaxArray& array)
    {
    switch (array.Type)
      {
    case DaxArray::IRREGULAR:
      return DaxArrayIrregular::GetVector4(work, array);
      }
    return make_DaxVector4(0, 0, 0, 0);
    }
};

#endif
