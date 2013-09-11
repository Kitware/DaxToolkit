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
//  Copyright 2012 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================

// This test makes sure that the algorithms specified in
// DeviceAdapterAlgorithmGeneral.h are working correctly. It does this by
// creating a test device adapter that uses the serial device adapter for the
// base schedule/scan/sort algorithms and using the general algorithms for
// everything else. Because this test is based of the serial device adapter,
// make sure that UnitTestDeviceAdapterSerial is working before trying to debug
// this one.

#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_ERROR

#include <dax/cont/DeviceAdapterSerial.h>
#include <dax/cont/internal/DeviceAdapterAlgorithmGeneral.h>

#include <dax/cont/testing/TestingDeviceAdapter.h>

namespace dax {
namespace cont {
namespace testing {

struct DeviceAdapterTagTestAlgorithmGeneral { };

}
}
}

namespace
{
template<class T>
struct testing_device_handle
{
  typedef dax::cont::ArrayHandle<T,
         dax::cont::ArrayContainerControlTagBasic,
         dax::cont::testing::DeviceAdapterTagTestAlgorithmGeneral> type;
};

template<class T>
struct serial_device_handle
{
  typedef dax::cont::ArrayHandle<T,
         dax::cont::ArrayContainerControlTagBasic,
         dax::cont::DeviceAdapterTagSerial> type;
};

template<class ValueType, class ContainerTag>
struct SpecialCopy
{
  dax::cont::ArrayHandle<ValueType,
        ContainerTag,dax::cont::DeviceAdapterTagSerial> Copy;

  template<typename T>
  DAX_CONT_EXPORT
  void Fill(T& t)
  {
  this->Copy.PrepareForOutput(t.GetNumberOfValues());
  t.CopyInto(this->Copy.GetPortalControl().GetIteratorBegin());
  }

  template<typename T>
  DAX_CONT_EXPORT
  void WriteBack(T& t)
  {
  this->Copy.CopyInto(t.GetPortalControl().GetIteratorBegin());
  }
};

template<class U, class V>
struct SpecialCopy< dax::Pair<U,V>,
  typename dax::cont::internal::ArrayContainerControlZipTypes<
        typename testing_device_handle<U>::type,
        typename testing_device_handle<V>::type >::ArrayContainerControlTag >
{
  typedef dax::Pair<U,V> ValueType;
  typedef typename serial_device_handle<U>::type KeyHandle;
  typedef typename serial_device_handle<V>::type ValueHandle;
  typedef typename dax::cont::internal::ArrayContainerControlZipTypes<
        KeyHandle, ValueHandle >::ArrayContainerControlTag ContainerTag;


  KeyHandle Keys;
  ValueHandle Values;
  dax::cont::ArrayHandle<ValueType,
        ContainerTag,dax::cont::DeviceAdapterTagSerial> Copy;

  template<typename T>
  DAX_CONT_EXPORT
  void Fill(T& t)
  {
  //we now need to copy the values of t into zip
  const dax::Id size = t.GetNumberOfValues();
  this->Keys.PrepareForOutput(size);
  this->Values.PrepareForOutput(size);
  for(dax::Id i=0; i < size; ++i)
    {
    this->Keys.GetPortalControl().Set(i, t.GetPortalControl().Get(i).first);
    this->Values.GetPortalControl().Set(i, t.GetPortalControl().Get(i).second);
    }
  this->Copy=dax::cont::internal::make_ArrayHandleZip(this->Keys,this->Values);
  }

  template<typename T>
  DAX_CONT_EXPORT
  void WriteBack(T& t)
  {
  const dax::Id size = this->Copy.GetNumberOfValues();
  for(dax::Id i=0; i < size; ++i)
    {
    t.GetPortalControl().Set(i, this->Copy.GetPortalControl().Get(i) );
    }
  }

};

}

namespace dax {
namespace cont {
namespace internal {

template <typename T, class ArrayContainerControlTag>
class ArrayManagerExecution
    <T,
    ArrayContainerControlTag,
    dax::cont::testing::DeviceAdapterTagTestAlgorithmGeneral>
    : public dax::cont::internal::ArrayManagerExecution
          <T, ArrayContainerControlTag, dax::cont::DeviceAdapterTagSerial>
{
public:
  typedef dax::cont::internal::ArrayManagerExecution
      <T, ArrayContainerControlTag, dax::cont::DeviceAdapterTagSerial>
      Superclass;
  typedef typename Superclass::ValueType ValueType;
  typedef typename Superclass::PortalType PortalType;
  typedef typename Superclass::PortalConstType PortalConstType;
};


template<>
struct DeviceAdapterAlgorithm<
           dax::cont::testing::DeviceAdapterTagTestAlgorithmGeneral> :
    dax::cont::internal::DeviceAdapterAlgorithmGeneral<
        DeviceAdapterAlgorithm<
                   dax::cont::testing::DeviceAdapterTagTestAlgorithmGeneral>,
        dax::cont::testing::DeviceAdapterTagTestAlgorithmGeneral>
{
private:
  typedef dax::cont::internal::DeviceAdapterAlgorithm<
      dax::cont::DeviceAdapterTagSerial> Algorithm;

  typedef dax::cont::testing::DeviceAdapterTagTestAlgorithmGeneral
            DeviceAdapterTagTestAlgorithmGeneral;

public:

  template<class Functor>
  DAX_CONT_EXPORT static void Schedule(Functor functor,
                                       dax::Id numInstances)
  {
    Algorithm::Schedule(functor, numInstances);
  }

  template<class Functor>
  DAX_CONT_EXPORT static void Schedule(Functor functor,
                                       dax::Id3 rangeMax)
  {
    Algorithm::Schedule(functor, rangeMax);
  }

  template<typename T, class CIn, class COut>
  DAX_CONT_EXPORT static T ScanExclusive(
      const dax::cont::ArrayHandle<T,CIn,DeviceAdapterTagTestAlgorithmGeneral> &input,
      dax::cont::ArrayHandle<T,COut,DeviceAdapterTagTestAlgorithmGeneral>& output)
  {
    // Need to use array handles compatible with serial adapter.
    dax::cont::ArrayHandle<T,CIn,DeviceAdapterTagSerial>
        inputCopy(input.GetPortalConstControl());
    dax::cont::ArrayHandle<T,COut,DeviceAdapterTagSerial> originalOutput;

    T result = Algorithm::ScanExclusive(inputCopy, originalOutput);

    // Copy data back into original
    originalOutput.CopyInto(output.GetPortalControl().GetIteratorBegin());

    return result;
  }

  template<typename T, class Container>
  DAX_CONT_EXPORT static void Sort(
      dax::cont::ArrayHandle<T,Container,DeviceAdapterTagTestAlgorithmGeneral>& values)
  {
    // Need to use an array handle compatible with the serial adapter.
    dax::cont::ArrayHandle<T,Container,DeviceAdapterTagSerial> valuesCopy;
    // Allocate memory in valuesCopy (inefficiently).
    valuesCopy.PrepareForOutput(values.GetNumberOfValues());
    values.CopyInto(valuesCopy.GetPortalControl().GetIteratorBegin());

    Algorithm::Sort(valuesCopy);

    // Copy data back into original
    valuesCopy.CopyInto(values.GetPortalControl().GetIteratorBegin());
  }

  template<typename T, class Container, class Compare>
  DAX_CONT_EXPORT static void Sort(
      dax::cont::ArrayHandle<T,Container,DeviceAdapterTagTestAlgorithmGeneral>& values,
      Compare comp)

  {
    //ArrayHandleZip doesn't implement CopyInto, and the default
    //constructor of it when it is sliced is invalid so we can't use
    //any of the standard ways to copy the data
    SpecialCopy<T,Container> copy_helper;
    copy_helper.Fill(values);
    Algorithm::Sort(copy_helper.Copy,comp);
    copy_helper.WriteBack(values);
  }


  DAX_CONT_EXPORT static void Synchronize()
  {
    Algorithm::Synchronize();
  }
};

}
}
} // namespace dax::cont::testing

int UnitTestDeviceAdapterAlgorithmGeneral(int, char *[])
{
  return dax::cont::testing::TestingDeviceAdapter
      <dax::cont::testing::DeviceAdapterTagTestAlgorithmGeneral>::Run();
}
