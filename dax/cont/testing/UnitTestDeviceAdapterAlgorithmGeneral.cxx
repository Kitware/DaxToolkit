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
