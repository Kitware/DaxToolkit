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

// This these checks the creation of Thrust iterators within an OpenMP device
// adapter. If given a basic array container control (or any control that uses
// pointers for iterators), there is special code to use a thrust::device_ptr
// as the iterator, which handles everything correctly. That is tested in lots
// of different ways elsewhere in the code. However, if the iterator is not a
// pointer, then the device adapter has to construct its own iterator that is
// compatible with the device. In particular, we want to make sure we can write
// into this array properly.

#define DAX_DEVICE_ADAPTER DAX_DEVICE_ADAPTER_ERROR

#include <dax/openmp/cont/DeviceAdapterOpenMP.h>

#include <dax/VectorTraits.h>

#include <dax/cont/ArrayContainerControl.h>
#include <dax/cont/ArrayContainerControlBasic.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ArrayPortal.h>
#include <dax/cont/Assert.h>
#include <dax/cont/internal/IteratorFromArrayPortal.h>
#include <dax/cont/VectorOperations.h>

#include <dax/cont/testing/Testing.h>

namespace {

// This is just a simple example of a container that uses real arrays but does
// not return pointers for iterators. This may also have practical usage, in
// which case it could be moved to the dax::cont namespace rather than this
// test source.
template<class ComponentContainerTag>
struct ArrayContainerControlTagZip {  };

} // anonymous namespace

namespace dax {
namespace cont {
namespace internal {

template<typename T, class ComponentArrayPortal>
class ArrayPortalZip
{
public:
  typedef T ValueType;
  static const int NUM_COMPONENTS =dax::VectorTraits<ValueType>::NUM_COMPONENTS;

  DAX_CONT_EXPORT ArrayPortalZip() {  }

  DAX_CONT_EXPORT
  ArrayPortalZip(const dax::Tuple<ComponentArrayPortal,NUM_COMPONENTS> &portals)
    : ComponentPortals(portals)
  {
#ifndef NDEBUG
    for (int index = 1; index < NUM_COMPONENTS; index++)
      {
      DAX_ASSERT_CONT(this->ComponentPortals[0].GetNumberOfValues()
                      == this->ComponentPortals[index].GetNumberOfValues());
      }
#endif //NDEBUG
  }

  DAX_CONT_EXPORT
  dax::Id GetNumberOfValues() const
  {
    return this->ComponentPortals[0].GetNumberOfValues();
  }

  DAX_CONT_EXPORT
  ValueType Get(dax::Id index) const
  {
    ValueType result;
    for (int componentIndex = 0;
         componentIndex < NUM_COMPONENTS;
         componentIndex++)
      {
      dax::VectorTraits<ValueType>::SetComponent(
            result,
            componentIndex,
            this->ComponentPortals[componentIndex].Get(index));
      }
    return result;
  }

  DAX_CONT_EXPORT
  void Set(dax::Id index, const ValueType &value) const
  {
    for (int componentIndex = 0;
         componentIndex < NUM_COMPONENTS;
         componentIndex++)
      {
      this->ComponentPortals[componentIndex].Set(
            index,
            dax::VectorTraits<ValueType>::GetComponent(value, componentIndex));
      }
  }

  typedef dax::cont::internal::IteratorFromArrayPortal<ArrayPortalZip
      <T,ComponentArrayPortal> > IteratorType;

  DAX_CONT_EXPORT
  IteratorType GetIteratorBegin() const
  {
    return IteratorType(*this);
  }

  DAX_CONT_EXPORT
  IteratorType GetIteratorEnd() const
  {
    return IteratorType(*this, this->GetNumberOfValues());
  }

private:
  dax::Tuple<ComponentArrayPortal, NUM_COMPONENTS> ComponentPortals;
};

template<typename T, class ComponentContainerTag>
class ArrayContainerControl<T, ArrayContainerControlTagZip<ComponentContainerTag> >
{
private:
  typedef typename dax::VectorTraits<T>::ComponentType ComponentType;
  typedef dax::cont::internal::ArrayContainerControl
      <ComponentType, ComponentContainerTag> ComponentContainerType;
  typedef typename ComponentContainerType::PortalType ComponentPortalType;
  typedef typename ComponentContainerType::PortalConstType
      ComponentPortalConstType;
public:
  typedef T ValueType;
  static const int NUM_COMPONENTS =dax::VectorTraits<ValueType>::NUM_COMPONENTS;
  typedef dax::cont::internal::ArrayPortalZip
      <ValueType, ComponentPortalType> PortalType;
  typedef dax::cont::internal::ArrayPortalZip
      <ValueType, ComponentPortalConstType> PortalConstType;

  DAX_CONT_EXPORT PortalType GetPortal()
  {
    dax::Tuple<ComponentPortalType, NUM_COMPONENTS> ComponentPortals;
    for (int componentIndex = 0;
         componentIndex < NUM_COMPONENTS;
         componentIndex++)
      {
      ComponentPortals[componentIndex]
          = this->ComponentContainers[componentIndex].GetPortal();
      }
    return PortalType(ComponentPortals);
  }

  DAX_CONT_EXPORT PortalConstType GetPortalConst() const
  {
    dax::Tuple<ComponentPortalConstType, NUM_COMPONENTS> ComponentPortals;
    for (int componentIndex = 0;
         componentIndex < NUM_COMPONENTS;
         componentIndex++)
      {
      ComponentPortals[componentIndex]
          = this->ComponentContainers[componentIndex].GetPortalConst();
      }
    return PortalConstType(ComponentPortals);
  }

  DAX_CONT_EXPORT dax::Id GetNumberOfValues() const
  {
    return this->ComponentContainers[0].GetNumberOfValues();
  }

private:
  struct AllocateFunctor {
    dax::Id NumberOfValues;
    AllocateFunctor(dax::Id numberOfValues) : NumberOfValues(numberOfValues) { }
    void operator()(ComponentContainerType &container) const {
      container.Allocate(this->NumberOfValues);
    }
  };
public:
  DAX_CONT_EXPORT void Allocate(dax::Id numberOfValues)
  {
    dax::cont::VectorForEach(this->ComponentContainers,
                             AllocateFunctor(numberOfValues));
  }

private:
  struct ShrinkFunctor {
    dax::Id NumberOfValues;
    ShrinkFunctor(dax::Id numberOfValues) : NumberOfValues(numberOfValues) { }
    void operator()(ComponentContainerType &container) const {
      container.Shrink(this->NumberOfValues);
    }
  };
public:
  DAX_CONT_EXPORT void Shrink(dax::Id numberOfValues)
  {
    dax::cont::VectorForEach(this->ComponentContainers,
                             ShrinkFunctor(numberOfValues));
  }

private:
  struct ReleaseResourcesFunctor {
    void operator()(ComponentContainerType &container) const {
      container.ReleaseResources();
    }
  };
public:
  DAX_CONT_EXPORT void ReleaseResources()
  {
    dax::cont::VectorForEach(this->ComponentContainers,
                             ReleaseResourcesFunctor());
  }

private:
  dax::Tuple<ComponentContainerType, NUM_COMPONENTS> ComponentContainers;
};

}
}
}

namespace {

void TestCustomContainer()
{
  const int ARRAY_SIZE = 10;
  const dax::Scalar TEST_VALUE = 1234.5678;

  std::cout << "Set up input array." << std::endl;
  dax::Vector2 inputBuffer[ARRAY_SIZE];
  for (dax::Id index = 0; index < ARRAY_SIZE; index++)
    {
    inputBuffer[index][0] = TEST_VALUE;
    inputBuffer[index][1] = index;
    }
  dax::cont::ArrayHandle<
      dax::Vector2,
      dax::cont::ArrayContainerControlTagBasic,
      dax::openmp::cont::DeviceAdapterTagOpenMP> inputArray =
      dax::cont::make_ArrayHandle(inputBuffer,
                                  ARRAY_SIZE,
                                  dax::cont::ArrayContainerControlTagBasic(),
                                  dax::openmp::cont::DeviceAdapterTagOpenMP());

  std::cout << "Set up output array." << std::endl;
  typedef dax::cont::ArrayHandle<
      dax::Vector2,
      ArrayContainerControlTagZip<dax::cont::ArrayContainerControlTagBasic>,
      dax::openmp::cont::DeviceAdapterTagOpenMP> OutputArrayType;
  OutputArrayType outputArray;

  std::cout << "Do a simple operation on the arrays." << std::endl;
  dax::cont::internal::DeviceAdapterAlgorithm<
      dax::openmp::cont::DeviceAdapterTagOpenMP>::Copy(inputArray, outputArray);

  std::cout << "Check the result." << std::endl;
  OutputArrayType::PortalConstControl outputPortal =
      outputArray.GetPortalConstControl();
  for (dax::Id index = 0; index < ARRAY_SIZE; index++)
    {
    dax::Vector2 value = outputPortal.Get(index);
    DAX_TEST_ASSERT(value[0] == TEST_VALUE, "Values not copied.");
    DAX_TEST_ASSERT(value[1] == index, "Values not copied");
    }

  std::cout << "Test passed." << std::endl;
}

} // anonymous namespace

int OpenMPCustomContainer(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestCustomContainer);
}
