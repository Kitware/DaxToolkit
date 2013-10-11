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
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapterSerial.h>
#include <dax/cont/ReduceKeysValues.h>

#include <dax/exec/WorkletReduceKeysValues.h>

#include <dax/cont/testing/Testing.h>

#include <algorithm>
#include <map>
#include <set>
#include <vector>

#include <stdlib.h>
#include <time.h>

#define PRINT_VALUES

namespace {

typedef dax::cont::ArrayContainerControlTagBasic Container;
typedef dax::cont::DeviceAdapterTagSerial DeviceAdapter;

typedef dax::cont::ArrayHandle<dax::Id,Container,DeviceAdapter> ArrayType;

const int ARRAY_SIZE = 10;
const int NUM_KEYS = 10;

struct DummyWorklet : dax::exec::WorkletReduceKeysValues {  };

typedef std::map<dax::Id, std::set<dax::Id> > KeyMapType;
typedef dax::cont::ReduceKeysValues<DummyWorklet, ArrayType> DaxKeyMapType;

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

#ifdef PRINT_VALUES
  for (KeyMapType::iterator mapIter = keyMap.begin();
       mapIter != keyMap.end();
       mapIter++)
    {
    std::cout << mapIter->first << ":";
    PrintArray(mapIter->second.begin(), mapIter->second.end());
    }
#endif

  return keyMap;
}

DaxKeyMapType BuildDaxKeyMap(const ArrayType &inputKeys)
{
  std::cout << "Building Dax version of key map" << std::endl;

  DaxKeyMapType keyMap(inputKeys);
  keyMap.BuildReductionMap();

#ifdef PRINT_VALUES
  std::cout << "Counts:" << std::endl;
  PrintArray(
        keyMap.GetReductionCounts().GetPortalConstControl().GetIteratorBegin(),
        keyMap.GetReductionCounts().GetPortalConstControl().GetIteratorEnd());
  std::cout << "Offsets:" << std::endl;
  PrintArray(
        keyMap.GetReductionOffsets().GetPortalConstControl().GetIteratorBegin(),
        keyMap.GetReductionOffsets().GetPortalConstControl().GetIteratorEnd());
  std::cout << "Indices:" << std::endl;
  PrintArray(
        keyMap.GetReductionIndices().GetPortalConstControl().GetIteratorBegin(),
        keyMap.GetReductionIndices().GetPortalConstControl().GetIteratorEnd());
#endif

  return keyMap;
}

void CheckKeyMap(KeyMapType serialMap, DaxKeyMapType daxMap)
{
  std::cout << "Comparing key maps" << std::endl;

  typedef ArrayType::PortalConstControl PortalType;
  PortalType keys = daxMap.GetKeys().GetPortalConstControl();
  PortalType counts = daxMap.GetReductionCounts().GetPortalConstControl();
  PortalType offsets = daxMap.GetReductionOffsets().GetPortalConstControl();
  PortalType indices = daxMap.GetReductionIndices().GetPortalConstControl();

  dax::Id inputSize = keys.GetNumberOfValues();

  std::vector<bool> foundIndices(inputSize);
  std::fill(foundIndices.begin(), foundIndices.end(), false);

  DAX_TEST_ASSERT(indices.GetNumberOfValues() == inputSize,
                  "Wrong number of indices.");
  DAX_TEST_ASSERT(counts.GetNumberOfValues() == (dax::Id)serialMap.size(),
                  "Wrong number of counts.");
  DAX_TEST_ASSERT(offsets.GetNumberOfValues() == (dax::Id)serialMap.size(),
                  "Wrong number of offsets.");

  // Although the ordering of the output shouldn't really matter, this test
  // assumes that both are in ascending order.
  KeyMapType::iterator serialIter = serialMap.begin();
  for (dax::Id outputIndex = 0;
       outputIndex < counts.GetNumberOfValues();
       outputIndex++)
    {
    DAX_TEST_ASSERT(serialIter != serialMap.end(),
                    "Dax made too many key buckets.");
    dax::Id key = serialIter->first;
    dax::Id count = counts.Get(outputIndex);
    dax::Id offset = offsets.Get(outputIndex);
    DAX_TEST_ASSERT(count == static_cast<dax::Id>(serialIter->second.size()),
                    "Dax key bucket does not agree.");
    for (dax::Id visitIndex = 0; visitIndex < count; visitIndex++)
      {
      dax::Id inputIndex = indices.Get(offset + visitIndex);
      DAX_TEST_ASSERT((inputIndex >= 0) && (inputIndex < inputSize),
                      "Got bad index");
      DAX_TEST_ASSERT(!foundIndices[inputIndex],
                      "An index was listed twice.");
      foundIndices[inputIndex] = true;
      dax::Id foundKey = keys.Get(inputIndex);
      DAX_TEST_ASSERT(key == foundKey, "Bucket has wrong key.");
      }
    serialIter++;
    }
  DAX_TEST_ASSERT(serialIter == serialMap.end(),
                  "Dax made too few key buckets.");
  for (dax::Id inputIndex = 0; inputIndex < inputSize; inputIndex++)
    {
    DAX_TEST_ASSERT(foundIndices[inputIndex],
                    "An index did not make it to the indices list.");
    }
}

void RunBuildReductionMap()
{
  srandom(time(NULL));

  ArrayType randomKeyInput = MakeInputArray();
  KeyMapType serialMap = BuildSerialKeyMap(randomKeyInput);
  DaxKeyMapType daxMap = BuildDaxKeyMap(randomKeyInput);
  CheckKeyMap(serialMap, daxMap);
}

} // anonymous namespace

int UnitTestBuildReductionMap(int, char *[])
{
  return dax::cont::testing::Testing::Run(RunBuildReductionMap);
}
