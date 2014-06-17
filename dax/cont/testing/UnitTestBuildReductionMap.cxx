//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2013 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================

#define DAX_ARRAY_CONTAINER_CONTROL DAX_ARRAY_CONTAINER_CONTROL_ERROR
#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_ERROR

#include <dax/cont/ArrayContainerControlBasic.h>
#include <dax/cont/ArrayHandleCounting.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapterSerial.h>
#include <dax/cont/DispatcherReduceKeysValues.h>

#include <dax/exec/WorkletReduceKeysValues.h>

#include <dax/cont/testing/Testing.h>

#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include <time.h>

// #define PRINT_VALUES

namespace {

typedef dax::cont::ArrayContainerControlTagBasic Container;
typedef dax::cont::DeviceAdapterTagSerial DeviceAdapter;

typedef dax::cont::ArrayHandle<dax::Id,Container,DeviceAdapter> ArrayType;

const int ARRAY_SIZE = 10;
const int NUM_KEYS = 10;

typedef std::map<dax::Id, std::set<dax::Id> > KeyMapType;

class TrackReduceWorklet : public dax::exec::WorkletReduceKeysValues
{
public:
  TrackReduceWorklet():
    KeyValues( new KeyMapType() )
  {}

  typedef void ControlSignature( ValuesIn );
  typedef void ExecutionSignature(KeyGroup(_1), WorkId);

  template<typename KeyGroupType>
  DAX_EXEC_EXPORT
  void operator()(KeyGroupType inPortal, dax::Id work_id) const
  {
  for(dax::Id iCtr = 0; iCtr < inPortal.GetNumberOfValues(); iCtr++)
    {
    (*this->KeyValues)[work_id].insert( inPortal[iCtr] );
    }

  }

  //only works with serial/openmp/tbb backends
  boost::shared_ptr<KeyMapType> KeyValues;
};




template<class IteratorType>
void PrintArray(IteratorType beginIter, IteratorType endIter)
{
  for (IteratorType iter = beginIter; iter != endIter; iter++)
    {
    std::cout << " " << *iter;
    }
  std::cout << std::endl;
}

ArrayType MakeInputArray()
{
  std::cout << "Building input array" << std::endl;

  std::vector<dax::Id> data(ARRAY_SIZE);
  for (std::vector<dax::Id>::iterator iter = data.begin();
       iter != data.end();
       iter++)
    {
    *iter = random()%NUM_KEYS;
    }

#ifdef PRINT_VALUES
  PrintArray(data.begin(), data.end());
#endif

  // Create an array handle from our data buffer.
  ArrayType dataWrapper =
      dax::cont::make_ArrayHandle(data,Container(),DeviceAdapter());

  // Make a copy of the data because the buffer is going to be deleted when
  // we go out of scope.  (Perhaps the deep array copy should be made more
  // accessable to client code.)
  ArrayType dataCopy;
  dax::cont::DeviceAdapterAlgorithm<DeviceAdapter>::Copy(dataWrapper,
                                                                   dataCopy);

  return dataCopy;
}

KeyMapType BuildSerialKeyMap(const ArrayType &inputKeys)
{
  std::cout << "Building key map in serial structure" << std::endl;
  KeyMapType keyMap;

  ArrayType::PortalConstControl keysPortal = inputKeys.GetPortalConstControl();
  for (dax::Id index = 0; index < keysPortal.GetNumberOfValues(); index++)
    {
    keyMap[keysPortal.Get(index)].insert(index);
    }

  return keyMap;
}

void CheckKeyMap(const ArrayType &inputKeys, KeyMapType serialMap)
{
  std::cout << "Run Dax KeyMap and store counts, offsets, and indices" << std::endl;

  TrackReduceWorklet track;
  dax::cont::DispatcherReduceKeysValues< TrackReduceWorklet,
        ArrayType, DeviceAdapter > reduceKeyValues(inputKeys, track);

  //we need to pass the index as the values to bind too
  typedef dax::cont::ArrayHandleCounting<dax::Id, DeviceAdapter> CountingHandle;

  reduceKeyValues.Invoke( CountingHandle(0,inputKeys.GetNumberOfValues()) );


  //compose the info we stored during execution to make up
  KeyMapType* daxKeyMap = track.KeyValues.get();

  std::cout << "Comparing key maps" << std::endl;
  DAX_TEST_ASSERT(daxKeyMap->size() == serialMap.size(),
                  "Wrong number of indices.");

  typedef KeyMapType::const_iterator it;
  it serElem = serialMap.begin();
  for( it daxElem = daxKeyMap->begin();
       daxElem != daxKeyMap->end();
       ++daxElem, ++serElem)
    {
    bool matches = std::equal( daxElem->second.begin(), daxElem->second.end(),
                               serElem->second.begin() );
    DAX_TEST_ASSERT( matches == true, "key index values are wrong.");
    }
}

void RunBuildReductionMap()
{
  srandom(time(NULL));

  ArrayType randomKeyInput = MakeInputArray();
  KeyMapType serialMap = BuildSerialKeyMap(randomKeyInput);
  CheckKeyMap(randomKeyInput, serialMap);
}

} // anonymous namespace

int UnitTestBuildReductionMap(int, char *[])
{
  return dax::cont::testing::Testing::Run(RunBuildReductionMap);
}
