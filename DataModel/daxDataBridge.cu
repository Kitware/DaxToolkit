/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxDataBridge.h"

#include "daxDataArray.h"
#include "DaxDataArray.h"
#include "daxDataSet.h"
#include "daxKernelArgument.h"

#include <thrust/host_vector.h>

#include <map>
#include <set>

namespace
{
  DaxDataSet Convert(daxDataSetPtr ds,
    std::map<daxDataArrayPtr, int> &arrayIndexes)
    {
    DaxDataSet temp;
    if (ds->GetPointCoordinates())
      {
      temp.PointCoordinatesIndex = arrayIndexes[ds->GetPointCoordinates()];
      }

    if (ds->GetCellArray())
      {
      temp.CellArrayIndex = arrayIndexes[ds->GetCellArray()];
      }

    int cc;
    std::vector<daxDataArrayPtr>::iterator iter2;
    for (cc=0, iter2 = ds->PointData.begin(); iter2 != ds->PointData.end();
      ++iter2, ++cc)
      {
      temp.PointDataIndices[cc] = arrayIndexes[*iter2];
      }

    for (cc=0, iter2 = ds->CellData.begin(); iter2 != ds->CellData.end();
      ++iter2, ++cc)
      {
      arrayIndexes[*iter2] = -1;
      temp.CellDataIndices[cc] = arrayIndexes[*iter2];
      }
    return temp;
    }

  void AddArrays(daxDataSetPtr ds, std::map<daxDataArrayPtr, int> &arrayIndexes, bool input)
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

    std::vector<daxDataArrayPtr>::iterator iter2;
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

};

class daxDataBridge::daxInternals
{
public:
  std::vector<daxDataSetPtr> Inputs;
  std::vector<daxDataSetPtr> Intermediates;
  std::vector<daxDataSetPtr> Outputs;
  std::map<daxDataArray*, int> Arrays;
};

//-----------------------------------------------------------------------------
daxDataBridge::daxDataBridge()
{
  this->Internals = new daxInternals();
}

//-----------------------------------------------------------------------------
daxDataBridge::~daxDataBridge()
{
  delete this->Internals;
  this->Internals = NULL;
}

//-----------------------------------------------------------------------------
void daxDataBridge::AddInputData(daxDataSetPtr dataset)
{
  this->Internals->Inputs.push_back(dataset);
}

//-----------------------------------------------------------------------------
void daxDataBridge::AddIntermediateData(daxDataSetPtr dataset)
{
  this->Internals->Intermediates.push_back(dataset);
}

//-----------------------------------------------------------------------------
void daxDataBridge::AddOutputData(daxDataSetPtr dataset)
{
  this->Internals->Outputs.push_back(dataset);
}

//-----------------------------------------------------------------------------
daxKernelArgumentPtr daxDataBridge::Upload() const
{
  // * Upload all daxDataArray's.
  std::vector<daxDataSetPtr>::iterator ds_iter;

  // First build the list of unique arrays we need to upload.
  std::map<daxDataArrayPtr, int> arrayIndexes;

  for (ds_iter = this->Internals->Inputs.begin();
    ds_iter != this->Internals->Inputs.end(); ++ds_iter)
    {
    daxDataSetPtr ds = *ds_iter;
    AddArrays(ds, arrayIndexes, true);
    }

  // FIXME: skipping intermediates for now, since I am not sure how to handle
  // them.

  for (ds_iter = this->Internals->Outputs.begin();
    ds_iter != this->Internals->Outputs.end(); ++ds_iter)
    {
    daxDataSetPtr ds = *ds_iter;
    AddArrays(ds, arrayIndexes, false);
    }

  daxKernelArgumentPtr argument(new daxKernelArgument);

  std::map<daxDataArrayPtr, int>::iterator map_iter;
  for (map_iter = arrayIndexes.begin(); map_iter != arrayIndexes.end();
    ++map_iter)
    {
    DaxDataArray temp;
    if (map_iter->first->Convert(&temp))
      {
      size_t index = argument->Arrays.size();
      DaxDataArray device_temp = temp;
      if (temp.RawData)
        {
        cudaMalloc(&device_temp.RawData, temp.SizeInBytes);
        }
      if (map_iter->second == -1)
        {
        cudaMemcpy(device_temp.RawData, temp.RawData,
          temp.SizeInBytes, cudaMemcpyHostToDevice);
        }

      argument->Arrays.push_back(device_temp);
      map_iter->second = index;
      }
    }

  // now that arrays have been uploaded, upload the datasets.
  for (ds_iter = this->Internals->Inputs.begin();
    ds_iter != this->Internals->Inputs.end(); ++ds_iter)
    {
    daxDataSetPtr ds = *ds_iter;
    DaxDataSet temp = Convert(ds, arrayIndexes);
    argument->Datasets.push_back(temp);
    }

  for (ds_iter = this->Internals->Intermediates.begin();
    ds_iter != this->Internals->Intermediates.end(); ++ds_iter)
    {
    daxDataSetPtr ds = *ds_iter;
    DaxDataSet temp = Convert(ds, arrayIndexes);
    argument->Datasets.push_back(temp);
    }

  for (ds_iter = this->Internals->Outputs.begin();
    ds_iter != this->Internals->Outputs.end(); ++ds_iter)
    {
    daxDataSetPtr ds = *ds_iter;
    DaxDataSet temp = Convert(ds, arrayIndexes);
    argument->Datasets.push_back(temp);
    }

  argument->ArrayMap = arrayIndexes;
  return argument;
}

//-----------------------------------------------------------------------------
bool daxDataBridge::Download(daxKernelArgumentPtr argument) const
{
//  // iterate over output datasets and download all arrays. For now, I'll only
//  // download the cell-data and point-data arrays.
//  for (ds_iter = this->Internals->Outputs.begin();
//    ds_iter != this->Internals->Outputs.end(); ++ds_iter)
//    {
//    daxDataSetPtr ds = *ds_iter;
//    }
  return true;
}
