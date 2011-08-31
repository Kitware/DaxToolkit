/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include <dax/cuda/cont/internal/DataBridge.h>

#include <dax/internal/DataArray.h>

#include <dax/cont/DataArray.h>
#include <dax/cont/DataArrayIrregular.h>
#include <dax/cont/DataArrayStructuredConnectivity.h>
#include <dax/cont/DataArrayStructuredPoints.h>
#include <dax/cont/DataSet.h>

#include <dax/cuda/cont/internal/DataArray.h>
#include <dax/cuda/cont/internal/KernelArgument.h>

#include <map>
#include <set>
#include <assert.h>

namespace {
  //-----------------------------------------------------------------------------
  dax::internal::DataSet Convert(dax::cont::DataSetPtr ds,
    std::map<dax::cont::DataArrayPtr, int> &arrayIndexes)
    {
    dax::internal::DataSet temp;
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
    dax::internal::DataArray UploadIrregular(
      dax::cont::DataArrayIrregular<T>* host_array, bool copy_heavy_data)
      {
      return copy_heavy_data?
        dax::cuda::cont::internal::CreateAndCopyToDevice(dax::internal::DataArray::IRREGULAR,
          dax::internal::DataArray::type(T()),
          sizeof(T) * host_array->GetHeavyData().size(),
          host_array->GetHeavyData().data()) :
        dax::cuda::cont::internal::CreateOnDevice(dax::internal::DataArray::IRREGULAR,
          dax::internal::DataArray::type(T()),
          sizeof(T) * host_array->GetHeavyData().size());
      }

  //-----------------------------------------------------------------------------
  template <class T>
    bool DownloadIrregular(
      dax::cont::DataArrayIrregular<T>* host_array,
      const dax::internal::DataArray& device_array)
      {
      assert (device_array.SizeInBytes == sizeof(T) * host_array->GetHeavyData().size());
      return dax::cuda::cont::internal::CopyToHost(device_array,
        host_array->GetHeavyData().data(), device_array.SizeInBytes);
      }

  //-----------------------------------------------------------------------------
  dax::internal::DataArray UploadStructuredPoints(
    dax::cont::DataArrayStructuredPoints* array, bool copy_heavy_data)
    {
    return copy_heavy_data?
      dax::cuda::cont::internal::CreateAndCopyToDevice(
        dax::internal::DataArray::STRUCTURED_POINTS,
        dax::internal::DataArray::VECTOR3,
        sizeof(dax::StructuredPointsMetaData),
        &array->GetHeavyData()) :
      dax::cuda::cont::internal::CreateOnDevice(
        dax::internal::DataArray::STRUCTURED_POINTS,
        dax::internal::DataArray::VECTOR3,
        sizeof(dax::StructuredPointsMetaData));
    }

  //-----------------------------------------------------------------------------
  dax::internal::DataArray UploadStructuredConnectivity(
    dax::cont::DataArrayStructuredConnectivity* array, bool copy_heavy_data)
    {
    return copy_heavy_data?
      dax::cuda::cont::internal::CreateAndCopyToDevice(
        dax::internal::DataArray::STRUCTURED_CONNECTIVITY,
        dax::internal::DataArray::VECTOR3,
        sizeof(dax::StructuredPointsMetaData),
        &array->GetHeavyData()) :
      dax::cuda::cont::internal::CreateOnDevice(
        dax::internal::DataArray::STRUCTURED_CONNECTIVITY,
        dax::internal::DataArray::VECTOR3,
        sizeof(dax::StructuredPointsMetaData));
    }
}

class dax::cuda::cont::internal::DataBridge::daxInternals
{
public:
  std::vector<dax::cont::DataSetPtr> Inputs;
  std::vector<dax::cont::DataSetPtr> Intermediates;
  std::vector<dax::cont::DataSetPtr> Outputs;
};

//-----------------------------------------------------------------------------
dax::cuda::cont::internal::DataBridge::DataBridge()
{
  this->Internals = new daxInternals();
}

//-----------------------------------------------------------------------------
dax::cuda::cont::internal::DataBridge::~DataBridge()
{
  delete this->Internals;
  this->Internals = NULL;
}

//-----------------------------------------------------------------------------
void dax::cuda::cont::internal::DataBridge::AddInputData(dax::cont::DataSetPtr dataset)
{
  this->Internals->Inputs.push_back(dataset);
}

//-----------------------------------------------------------------------------
void dax::cuda::cont::internal::DataBridge::AddIntermediateData(dax::cont::DataSetPtr dataset)
{
  this->Internals->Intermediates.push_back(dataset);
}

//-----------------------------------------------------------------------------
void dax::cuda::cont::internal::DataBridge::AddOutputData(dax::cont::DataSetPtr dataset)
{
  this->Internals->Outputs.push_back(dataset);
}

//-----------------------------------------------------------------------------
dax::cuda::cont::internal::KernelArgumentPtr dax::cuda::cont::internal::DataBridge::Upload() const
{
  // * Upload all dax::core::DataArray's.
  std::vector<dax::cont::DataSetPtr>::iterator ds_iter;

  // First build the list of unique arrays we need to upload.
  std::map<dax::cont::DataArrayPtr, int> arrayIndexes;

  for (ds_iter = this->Internals->Inputs.begin();
    ds_iter != this->Internals->Inputs.end(); ++ds_iter)
    {
    dax::cont::DataSetPtr ds = *ds_iter;
    AddArrays(ds, arrayIndexes, true);
    }

  for (ds_iter = this->Internals->Intermediates.begin();
    ds_iter != this->Internals->Intermediates.end(); ++ds_iter)
    {
    dax::cont::DataSetPtr ds = *ds_iter;
    AddArrays(ds, arrayIndexes, false);
    }

  for (ds_iter = this->Internals->Outputs.begin();
    ds_iter != this->Internals->Outputs.end(); ++ds_iter)
    {
    dax::cont::DataSetPtr ds = *ds_iter;
    AddArrays(ds, arrayIndexes, false);
    }


  std::vector<dax::internal::DataArray> arrays;
  int index = 0;
  std::map<dax::cont::DataArrayPtr, int>::iterator map_iter;
  for (map_iter = arrayIndexes.begin(); map_iter != arrayIndexes.end();
    ++map_iter)
    {
    dax::internal::DataArray temp = this->UploadArray(map_iter->first,
      /*copy_heavy_data=*/ map_iter->second == -1);
    arrays.push_back(temp);
    map_iter->second = index;
    index++;
    }

  std::vector<dax::internal::DataSet> datasets;

  // now that arrays have been uploaded, upload the datasets.
  for (ds_iter = this->Internals->Inputs.begin();
    ds_iter != this->Internals->Inputs.end(); ++ds_iter)
    {
    dax::cont::DataSetPtr ds = *ds_iter;
    dax::internal::DataSet temp = Convert(ds, arrayIndexes);
    datasets.push_back(temp);
    }

  for (ds_iter = this->Internals->Intermediates.begin();
    ds_iter != this->Internals->Intermediates.end(); ++ds_iter)
    {
    dax::cont::DataSetPtr ds = *ds_iter;
    dax::internal::DataSet temp = Convert(ds, arrayIndexes);
    datasets.push_back(temp);
    }

  for (ds_iter = this->Internals->Outputs.begin();
    ds_iter != this->Internals->Outputs.end(); ++ds_iter)
    {
    dax::cont::DataSetPtr ds = *ds_iter;
    dax::internal::DataSet temp = Convert(ds, arrayIndexes);
    datasets.push_back(temp);
    }

  dax::cuda::cont::internal::KernelArgumentPtr argument(new dax::cuda::cont::internal::KernelArgument);
  argument->SetArrays(arrays);
  argument->SetDataSets(datasets);
  argument->SetArrayMap(arrayIndexes);
  return argument;
}

//-----------------------------------------------------------------------------
bool dax::cuda::cont::internal::DataBridge::Download(dax::cuda::cont::internal::KernelArgumentPtr argument) const
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
        dax::internal::DataArray array = argument->HostArrays[argument->ArrayMap[*iter2]];
        this->DownloadArray(*iter2, array);
        }
      }
    for (iter2 = ds->CellData.begin(); iter2 != ds->CellData.end(); ++iter2)
      {
      if (argument->ArrayMap.find(*iter2) != argument->ArrayMap.end())
        {
        dax::internal::DataArray array = argument->HostArrays[argument->ArrayMap[*iter2]];
        this->DownloadArray(*iter2, array);
        }
      }

    }
  return true;
}

//-----------------------------------------------------------------------------
dax::internal::DataArray
dax::cuda::cont::internal::DataBridge::UploadArray(
  dax::cont::DataArrayPtr host_array, bool copy_heavy_data) const
{
  dax::cont::DataArrayScalar* scalar_ptr =
    dynamic_cast<dax::cont::DataArrayScalar*>(host_array.get());
  if (scalar_ptr!= NULL)
    {
    return UploadIrregular<dax::Scalar>(scalar_ptr,
      copy_heavy_data);
    }

  dax::cont::DataArrayVector3* v3_ptr =
    dynamic_cast<dax::cont::DataArrayVector3*>(host_array.get());
  if (v3_ptr!= NULL)
    {
    return UploadIrregular<dax::Vector3>(v3_ptr,
      copy_heavy_data);
    }

  dax::cont::DataArrayVector4* v4_ptr =
    dynamic_cast<dax::cont::DataArrayVector4*>(host_array.get());
  if (v4_ptr!= NULL)
    {
    return UploadIrregular<dax::Vector4>(v4_ptr,
      copy_heavy_data);
    }

  dax::cont::DataArrayStructuredConnectivity* conn_ptr =
    dynamic_cast<dax::cont::DataArrayStructuredConnectivity*>(host_array.get());
  if (conn_ptr)
    {
    return UploadStructuredConnectivity(conn_ptr, copy_heavy_data);
    }

  dax::cont::DataArrayStructuredPoints* points_ptr =
    dynamic_cast<dax::cont::DataArrayStructuredPoints*>(host_array.get());
  if (points_ptr)
    {
    return UploadStructuredPoints(points_ptr, copy_heavy_data);
    }

  // unhandled case.
  abort();
  return dax::internal::DataArray();
}

//-----------------------------------------------------------------------------
bool dax::cuda::cont::internal::DataBridge::DownloadArray(
  dax::cont::DataArrayPtr host_array, const dax::internal::DataArray& device_array) const
{
  // need to add logic to upload/download.
  dax::cont::DataArrayScalar* scalar_ptr =
    dynamic_cast<dax::cont::DataArrayScalar*>(host_array.get());
  if (scalar_ptr!= NULL)
    {
    return DownloadIrregular<dax::Scalar>(scalar_ptr,
      device_array);
    }

  dax::cont::DataArrayVector3* v3_ptr =
    dynamic_cast<dax::cont::DataArrayVector3*>(host_array.get());
  if (v3_ptr!= NULL)
    {
    return DownloadIrregular<dax::Vector3>(v3_ptr, device_array);
    }

  dax::cont::DataArrayVector4* v4_ptr =
    dynamic_cast<dax::cont::DataArrayVector4*>(host_array.get());
  if (v4_ptr!= NULL)
    {
    return DownloadIrregular<dax::Vector4>(v4_ptr, device_array);
    }

  // unhandled case.
  abort();
  return false;
}
