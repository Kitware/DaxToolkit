/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "CUDA/Control/DataBridge.h"

#include "Core/Common/DataArray.h"
#include "Interface/Control/DataArray.h"
#include "Interface/Control/DataArrayIrregular.h"
#include "Interface/Control/DataArrayStructuredConnectivity.h"
#include "Interface/Control/DataArrayStructuredPoints.h"
#include "Interface/Control/DataSet.h"
#include "CUDA/Control/DataArray.h"
#include "CUDA/Control/KernelArgument.h"

#include <map>
#include <set>
#include <assert.h>

namespace dax { namespace internal {
  //-----------------------------------------------------------------------------
  dax::core::DataSet Convert(dax::cont::DataSetPtr ds,
    std::map<dax::cont::DataArrayPtr, int> &arrayIndexes)
    {
    dax::core::DataSet temp;
    if (ds->GetPointCoordinates())
      {
      temp.PointCoordinatesIndex = arrayIndexes[ds->GetPointCoordinates()];
      }

    if (ds->GetCellArray())
      {
      temp.CellArrayIndex = arrayIndexes[ds->GetCellArray()];
      }

    int cc;
    std::vector<dax::cont::DataArrayPtr>::iterator iter2;
    for (cc=0, iter2 = ds->PointData.begin(); iter2 != ds->PointData.end();
      ++iter2, ++cc)
      {
      temp.PointDataIndices[cc] = arrayIndexes[*iter2];
      }

    for (cc=0, iter2 = ds->CellData.begin(); iter2 != ds->CellData.end();
      ++iter2, ++cc)
      {
      temp.CellDataIndices[cc] = arrayIndexes[*iter2];
      }
    return temp;
    }

  //-----------------------------------------------------------------------------
  void AddArrays(dax::cont::DataSetPtr ds, std::map<dax::cont::DataArrayPtr, int> &arrayIndexes, 
    bool input)
    {
    if (ds->GetPointCoordinates())
      {
      if (arrayIndexes.find(ds->GetPointCoordinates()) == arrayIndexes.end())
        {
        arrayIndexes[ds->GetPointCoordinates()] = input? -1 : -2;
        }
      }
    if (ds->GetCellArray())
      {
      if (arrayIndexes.find(ds->GetCellArray()) == arrayIndexes.end())
        {
        arrayIndexes[ds->GetCellArray()] = input? -1 : -2;
        }
      }

    std::vector<dax::cont::DataArrayPtr>::iterator iter2;
    for (iter2 = ds->PointData.begin(); iter2 != ds->PointData.end(); ++iter2)
      {
      if (arrayIndexes.find(*iter2) == arrayIndexes.end())
        {
        arrayIndexes[*iter2] = input? -1 : -2;
        }
      }
    for (iter2 = ds->CellData.begin(); iter2 != ds->CellData.end(); ++iter2)
      {
      if (arrayIndexes.find(*iter2) == arrayIndexes.end())
        {
        arrayIndexes[*iter2] = input? -1 : -2;
        }
      }
    }


  //-----------------------------------------------------------------------------
  template <class T>
    dax::core::DataArray UploadIrregular(
      dax::cont::DataArrayIrregular<T>* host_array, bool copy_heavy_data)
      {
      return copy_heavy_data?
        dax::cuda::cont::CreateAndCopyToDevice(dax::core::DataArray::IRREGULAR,
          dax::core::DataArray::type(T()),
          sizeof(T) * host_array->GetHeavyData().size(),
          host_array->GetHeavyData().data()) :
        dax::cuda::cont::CreateOnDevice(dax::core::DataArray::IRREGULAR,
          dax::core::DataArray::type(T()),
          sizeof(T) * host_array->GetHeavyData().size());
      }

  //-----------------------------------------------------------------------------
  template <class T>
    bool DownloadIrregular(
      dax::cont::DataArrayIrregular<T>* host_array,
      const dax::core::DataArray& device_array)
      {
      assert (device_array.SizeInBytes == sizeof(T) * host_array->GetHeavyData().size());
      return dax::cuda::cont::CopyToHost(device_array,
        host_array->GetHeavyData().data(), device_array.SizeInBytes);
      }

  //-----------------------------------------------------------------------------
  dax::core::DataArray UploadStructuredPoints(
    dax::cont::DataArrayStructuredPoints* array, bool copy_heavy_data)
    {
    return copy_heavy_data?
      dax::cuda::cont::CreateAndCopyToDevice(
        dax::core::DataArray::STRUCTURED_POINTS,
        dax::core::DataArray::VECTOR3,
        sizeof(dax::StructuredPointsMetaData),
        &array->GetHeavyData()) :
      dax::cuda::cont::CreateOnDevice(
        dax::core::DataArray::STRUCTURED_POINTS,
        dax::core::DataArray::VECTOR3,
        sizeof(dax::StructuredPointsMetaData));
    }

  //-----------------------------------------------------------------------------
  dax::core::DataArray UploadStructuredConnectivity(
    dax::cont::DataArrayStructuredConnectivity* array, bool copy_heavy_data)
    {
    return copy_heavy_data?
      dax::cuda::cont::CreateAndCopyToDevice(
        dax::core::DataArray::STRUCTURED_CONNECTIVITY,
        dax::core::DataArray::VECTOR3,
        sizeof(dax::StructuredPointsMetaData),
        &array->GetHeavyData()) :
      dax::cuda::cont::CreateOnDevice(
        dax::core::DataArray::STRUCTURED_CONNECTIVITY,
        dax::core::DataArray::VECTOR3,
        sizeof(dax::StructuredPointsMetaData));
    }
}}

class dax::cuda::cont::DataBridge::daxInternals
{
public:
  std::vector<dax::cont::DataSetPtr> Inputs;
  std::vector<dax::cont::DataSetPtr> Intermediates;
  std::vector<dax::cont::DataSetPtr> Outputs;
};

//-----------------------------------------------------------------------------
dax::cuda::cont::DataBridge::DataBridge()
{
  this->Internals = new daxInternals();
}

//-----------------------------------------------------------------------------
dax::cuda::cont::DataBridge::~DataBridge()
{
  delete this->Internals;
  this->Internals = NULL;
}

//-----------------------------------------------------------------------------
void dax::cuda::cont::DataBridge::AddInputData(dax::cont::DataSetPtr dataset)
{
  this->Internals->Inputs.push_back(dataset);
}

//-----------------------------------------------------------------------------
void dax::cuda::cont::DataBridge::AddIntermediateData(dax::cont::DataSetPtr dataset)
{
  this->Internals->Intermediates.push_back(dataset);
}

//-----------------------------------------------------------------------------
void dax::cuda::cont::DataBridge::AddOutputData(dax::cont::DataSetPtr dataset)
{
  this->Internals->Outputs.push_back(dataset);
}

//-----------------------------------------------------------------------------
dax::cuda::cont::KernelArgumentPtr dax::cuda::cont::DataBridge::Upload() const
{
  // * Upload all dax::core::DataArray's.
  std::vector<dax::cont::DataSetPtr>::iterator ds_iter;

  // First build the list of unique arrays we need to upload.
  std::map<dax::cont::DataArrayPtr, int> arrayIndexes;

  for (ds_iter = this->Internals->Inputs.begin();
    ds_iter != this->Internals->Inputs.end(); ++ds_iter)
    {
    dax::cont::DataSetPtr ds = *ds_iter;
    dax::internal::AddArrays(ds, arrayIndexes, true);
    }

  for (ds_iter = this->Internals->Intermediates.begin();
    ds_iter != this->Internals->Intermediates.end(); ++ds_iter)
    {
    dax::cont::DataSetPtr ds = *ds_iter;
    dax::internal::AddArrays(ds, arrayIndexes, false);
    }

  for (ds_iter = this->Internals->Outputs.begin();
    ds_iter != this->Internals->Outputs.end(); ++ds_iter)
    {
    dax::cont::DataSetPtr ds = *ds_iter;
    dax::internal::AddArrays(ds, arrayIndexes, false);
    }


  std::vector<dax::core::DataArray> arrays;
  int index = 0;
  std::map<dax::cont::DataArrayPtr, int>::iterator map_iter;
  for (map_iter = arrayIndexes.begin(); map_iter != arrayIndexes.end();
    ++map_iter)
    {
    dax::core::DataArray temp = this->UploadArray(map_iter->first,
      /*copy_heavy_data=*/ map_iter->second == -1);
    arrays.push_back(temp);
    map_iter->second = index;
    index++;
    }

  std::vector<dax::core::DataSet> datasets;

  // now that arrays have been uploaded, upload the datasets.
  for (ds_iter = this->Internals->Inputs.begin();
    ds_iter != this->Internals->Inputs.end(); ++ds_iter)
    {
    dax::cont::DataSetPtr ds = *ds_iter;
    dax::core::DataSet temp = dax::internal::Convert(ds, arrayIndexes);
    datasets.push_back(temp);
    }

  for (ds_iter = this->Internals->Intermediates.begin();
    ds_iter != this->Internals->Intermediates.end(); ++ds_iter)
    {
    dax::cont::DataSetPtr ds = *ds_iter;
    dax::core::DataSet temp = dax::internal::Convert(ds, arrayIndexes);
    datasets.push_back(temp);
    }

  for (ds_iter = this->Internals->Outputs.begin();
    ds_iter != this->Internals->Outputs.end(); ++ds_iter)
    {
    dax::cont::DataSetPtr ds = *ds_iter;
    dax::core::DataSet temp = dax::internal::Convert(ds, arrayIndexes);
    datasets.push_back(temp);
    }

  dax::cuda::cont::KernelArgumentPtr argument(new dax::cuda::cont::KernelArgument);
  argument->SetArrays(arrays);
  argument->SetDataSets(datasets);
  argument->SetArrayMap(arrayIndexes);
  return argument;
}

//-----------------------------------------------------------------------------
bool dax::cuda::cont::DataBridge::Download(dax::cuda::cont::KernelArgumentPtr argument) const
{
  // iterate over output datasets and download all arrays. For now, I'll only
  // download the cell-data and point-data arrays.
  std::vector<dax::cont::DataSetPtr>::iterator ds_iter;
  for (ds_iter = this->Internals->Outputs.begin();
    ds_iter != this->Internals->Outputs.end(); ++ds_iter)
    {
    dax::cont::DataSetPtr ds = *ds_iter;

    std::vector<dax::cont::DataArrayPtr>::iterator iter2;
    for (iter2 = ds->PointData.begin(); iter2 != ds->PointData.end(); ++iter2)
      {
      if (argument->ArrayMap.find(*iter2) != argument->ArrayMap.end())
        {
        dax::core::DataArray array = argument->HostArrays[argument->ArrayMap[*iter2]];
        this->DownloadArray(*iter2, array);
        }
      }
    for (iter2 = ds->CellData.begin(); iter2 != ds->CellData.end(); ++iter2)
      {
      if (argument->ArrayMap.find(*iter2) != argument->ArrayMap.end())
        {
        dax::core::DataArray array = argument->HostArrays[argument->ArrayMap[*iter2]];
        this->DownloadArray(*iter2, array);
        }
      }

    }
  return true;
}

//-----------------------------------------------------------------------------
dax::core::DataArray
dax::cuda::cont::DataBridge::UploadArray(
  dax::cont::DataArrayPtr host_array, bool copy_heavy_data) const
{
  dax::cont::DataArrayScalar* scalar_ptr =
    dynamic_cast<dax::cont::DataArrayScalar*>(host_array.get());
  if (scalar_ptr!= NULL)
    {
    return dax::internal::UploadIrregular<dax::Scalar>(scalar_ptr,
      copy_heavy_data);
    }

  dax::cont::DataArrayVector3* v3_ptr =
    dynamic_cast<dax::cont::DataArrayVector3*>(host_array.get());
  if (v3_ptr!= NULL)
    {
    return dax::internal::UploadIrregular<dax::Vector3>(v3_ptr,
      copy_heavy_data);
    }

  dax::cont::DataArrayVector4* v4_ptr =
    dynamic_cast<dax::cont::DataArrayVector4*>(host_array.get());
  if (v4_ptr!= NULL)
    {
    return dax::internal::UploadIrregular<dax::Vector4>(v4_ptr,
      copy_heavy_data);
    }

  dax::cont::DataArrayStructuredConnectivity* conn_ptr =
    dynamic_cast<dax::cont::DataArrayStructuredConnectivity*>(host_array.get());
  if (conn_ptr)
    {
    return dax::internal::UploadStructuredConnectivity(conn_ptr, copy_heavy_data);
    }

  dax::cont::DataArrayStructuredPoints* points_ptr =
    dynamic_cast<dax::cont::DataArrayStructuredPoints*>(host_array.get());
  if (points_ptr)
    {
    return dax::internal::UploadStructuredPoints(points_ptr, copy_heavy_data);
    }

  // unhandled case.
  abort();
  return dax::core::DataArray();
}

//-----------------------------------------------------------------------------
bool dax::cuda::cont::DataBridge::DownloadArray(
  dax::cont::DataArrayPtr host_array, const dax::core::DataArray& device_array) const
{
  // need to add logic to upload/download.
  dax::cont::DataArrayScalar* scalar_ptr =
    dynamic_cast<dax::cont::DataArrayScalar*>(host_array.get());
  if (scalar_ptr!= NULL)
    {
    return dax::internal::DownloadIrregular<dax::Scalar>(scalar_ptr,
      device_array);
    }

  dax::cont::DataArrayVector3* v3_ptr =
    dynamic_cast<dax::cont::DataArrayVector3*>(host_array.get());
  if (v3_ptr!= NULL)
    {
    return dax::internal::DownloadIrregular<dax::Vector3>(v3_ptr, device_array);
    }

  dax::cont::DataArrayVector4* v4_ptr =
    dynamic_cast<dax::cont::DataArrayVector4*>(host_array.get());
  if (v4_ptr!= NULL)
    {
    return dax::internal::DownloadIrregular<dax::Vector4>(v4_ptr, device_array);
    }

  // unhandled case.
  abort();
  return false;
}
