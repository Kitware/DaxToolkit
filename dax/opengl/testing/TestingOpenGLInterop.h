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
#ifndef __dax__opengl__testing__TestingOpenGLInterop_h
#define __dax__opengl__testing__TestingOpenGLInterop_h

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DispatcherMapField.h>
#include <dax/worklet/Magnitude.h>

#include <dax/opengl/testing/TestingWindow.h>
#include <dax/opengl/TransferToOpenGL.h>

#include <dax/cont/testing/Testing.h>
#include <dax/cont/testing/TestingGridGenerator.h>

#include <algorithm>
#include <iterator>
#include <vector>


namespace dax {
namespace opengl {
namespace testing {

/// This class has a single static member, Run, that tests the templated
/// DeviceAdapter for support for opengl interop.
///
template< class DeviceAdapterTag,
          class ArrayContainerTag = DAX_DEFAULT_ARRAY_CONTAINER_CONTROL_TAG>
struct TestingOpenGLInterop
{
private:
  //fill the array with a collection of values and return it wrapped in
  //an dax array handle
  template<typename T>
  static
  dax::cont::ArrayHandle<T,ArrayContainerTag,DeviceAdapterTag>
  FillArray(std::vector<T>& data, std::size_t length)
  {
    typedef typename std::vector<T>::iterator iterator;
    //make sure the data array is exactly the right length
    data.clear();
    data.resize(length);
    dax::Id pos = 0;
    for(iterator i = data.begin(); i != data.end(); ++i, ++pos)
      { *i=T(pos); }

    std::random_shuffle(data.begin(),data.end());
    return dax::cont::make_ArrayHandle(data,
                                       ArrayContainerTag(),
                                       DeviceAdapterTag());
  }

  //Transfer the data in a dax ArrayHandle to open gl while making sure
  //we don't throw any errors
  template<typename ArrayHandleType>
  static
  void SafelyTransferArray(ArrayHandleType array, GLuint& handle)
  {
    try
      {
      dax::opengl::TransferToOpenGL(array,handle);
      }
    catch (dax::cont::ErrorControlOutOfMemory error)
      {
      std::cout << error.GetMessage() << std::endl;
      DAX_TEST_ASSERT(true==false,
                "Got an unexpected Out Of Memory error transferring to openGL");
      }
    catch (dax::cont::ErrorControlBadValue bvError)
      {
      std::cout << bvError.GetMessage() << std::endl;
      DAX_TEST_ASSERT(true==false,
                "Got an unexpected Bad Value error transferring to openGL");
      }
  }

  template<typename ArrayHandleType>
  static
  void SafelyTransferArray(ArrayHandleType array, GLuint& handle, GLenum type)
  {
    try
      {
      dax::opengl::TransferToOpenGL(array,handle,type);
      }
    catch (dax::cont::ErrorControlOutOfMemory error)
      {
      std::cout << error.GetMessage() << std::endl;
      DAX_TEST_ASSERT(true==false,
                "Got an unexpected Out Of Memory error transferring to openGL");
      }
    catch (dax::cont::ErrorControlBadValue bvError)
      {
      std::cout << bvError.GetMessage() << std::endl;
      DAX_TEST_ASSERT(true==false,
                "Got an unexpected Bad Value error transferring to openGL");
      }
  }



  //bring the data back from openGL and into a std vector. Will bind the
  //passed in handle to the default buffer type for the type T
  template<typename T>
  static
  std::vector<T> CopyGLBuffer(GLuint& handle, T t)
  {
    //get the type we used for this buffer.
    GLenum type = dax::opengl::internal::BufferTypePicker(t);

    //bind the buffer to the guessed buffer type, this way
    //we can call CopyGLBuffer no matter what it the active buffer
    glBindBuffer(type, handle);

    //get the size of the buffer
    int bytesInBuffer = 0;
    glGetBufferParameteriv(type, GL_BUFFER_SIZE, &bytesInBuffer);
    int size = ( bytesInBuffer / sizeof(T) );

    //get the buffer contents and place it into a vector
    std::vector<T> data;
    data.resize(size);
    glGetBufferSubData(type,0,bytesInBuffer,&data[0]);

    return data;
  }

  //make a random value that we can test when loading constant values
  template<typename T>
  static
  T MakeRandomValue(T)
  {
  return T(rand());
  }


  struct TransferFunctor
  {
    std::size_t Size;
    GLuint GLHandle;

    template <typename T>
    void operator()(const T t)
    {
      this->Size = 10;
      //verify that T is able to be transfer to openGL.
      //than pull down the results from the array buffer and verify
      //that they match the handles contents
      std::vector<T> tempData;
      dax::cont::ArrayHandle<T,ArrayContainerTag, DeviceAdapterTag> temp =
            FillArray(tempData,this->Size);

      //verify that the signature that doesn't have type works
      SafelyTransferArray(temp,GLHandle);

      bool  is_buffer;
      is_buffer = glIsBuffer(this->GLHandle);
      DAX_TEST_ASSERT(is_buffer==true,
                    "OpenGL buffer not filled");

      std::vector<T> returnedValues = CopyGLBuffer(this->GLHandle, t);

      //verify the results match what is in the array handle
      std::vector<T> expectedValues(returnedValues.size());
      temp.CopyInto( expectedValues.begin() );
      for(std::size_t i=0; i < this->Size; ++i)
        {
        DAX_TEST_ASSERT(test_equal(expectedValues[i],returnedValues[i]),
                        "Array Handle failed to transfer properly");
        }

      temp.ReleaseResources();
      temp = FillArray(tempData,this->Size*2);
      GLenum type = dax::opengl::internal::BufferTypePicker(t);
      SafelyTransferArray(temp,GLHandle,type);
      is_buffer = glIsBuffer(this->GLHandle);
      DAX_TEST_ASSERT(is_buffer==true,
                    "OpenGL buffer not filled");
      returnedValues = CopyGLBuffer(this->GLHandle, t);
      //verify the results match what is in the array handle
      expectedValues.resize(returnedValues.size());
      temp.CopyInto( expectedValues.begin() );
      for(std::size_t i=0; i < this->Size*2; ++i)
        {
        DAX_TEST_ASSERT(test_equal(expectedValues[i],returnedValues[i]),
                        "Array Handle failed to transfer properly");
        }


      //verify this work for a constant value array handle
      T constantValue = MakeRandomValue(t);
      dax::cont::ArrayHandleConstant<T,DeviceAdapterTag> constant(constantValue,
                                                                  this->Size);
      SafelyTransferArray(constant,GLHandle);
      is_buffer = glIsBuffer(this->GLHandle);
      DAX_TEST_ASSERT(is_buffer==true,
                    "OpenGL buffer not filled");
      returnedValues = CopyGLBuffer(this->GLHandle, constantValue);
      for(std::size_t i=0; i < this->Size; ++i)
        {
        DAX_TEST_ASSERT(test_equal(returnedValues[i],constantValue),
                        "Constant value array failed to transfer properly");
        }
    }
  };

  struct TransferGridFunctor
  {
    GLuint CoordGLHandle;
    GLuint MagnitudeGLHandle;

    template <typename GridType>
    void operator()(const GridType)
    {
    //verify we are able to be transfer both coordinates and indices to openGL.
    //than pull down the results from the array buffer and verify
    //that they match the handles contents
    dax::cont::testing::TestGrid<GridType,
                                 ArrayContainerTag,
                                 DeviceAdapterTag> grid(64);

    dax::cont::ArrayHandle<dax::Scalar,
                           ArrayContainerTag,
                           DeviceAdapterTag> magnitudeHandle;

    dax::cont::DispatcherMapField< dax::worklet::Magnitude,
                                   DeviceAdapterTag> dispatcher;
    dispatcher.Invoke(grid->GetPointCoordinates(), magnitudeHandle);

    //transfer to openGL 3 handles and catch any errors
    //
    SafelyTransferArray(grid->GetPointCoordinates(),this->CoordGLHandle);
    SafelyTransferArray(magnitudeHandle,this->MagnitudeGLHandle);

    //verify all 3 handles are actually handles
    bool  is_buffer = glIsBuffer(this->CoordGLHandle);
    DAX_TEST_ASSERT(is_buffer==true,
                    "Coordinates OpenGL buffer not filled");

    is_buffer = glIsBuffer(this->MagnitudeGLHandle);
    DAX_TEST_ASSERT(is_buffer==true,
                    "Magnitude OpenGL buffer not filled");

    //now that everything is openGL we have one task left.
    //transfer everything back to the host and compare it to the
    //computed values.
    std::vector<dax::Vector3> GLReturnedCoords = CopyGLBuffer(
                                        this->CoordGLHandle, dax::Vector3());
    std::vector<dax::Scalar> GLReturneMags = CopyGLBuffer(
                                        this->MagnitudeGLHandle,dax::Scalar());

    for (dax::Id pointIndex = 0;
         pointIndex < grid->GetNumberOfPoints();
         pointIndex++)
      {
      dax::Vector3 pointCoordinateExpected = grid.GetPointCoordinates(
                                                                    pointIndex);
      dax::Vector3 pointCoordinatesReturned =  GLReturnedCoords[pointIndex];
      DAX_TEST_ASSERT(test_equal(pointCoordinateExpected,
                                 pointCoordinatesReturned),
                      "Got bad coordinate from OpenGL buffer.");

      dax::Scalar magnitudeValue = GLReturneMags[pointIndex];
      dax::Scalar magnitudeExpected =
          sqrt(dax::dot(pointCoordinateExpected, pointCoordinateExpected));
      DAX_TEST_ASSERT(test_equal(magnitudeValue, magnitudeExpected),
                      "Got bad magnitude from OpenGL buffer.");
      }
    }
  };


public:
  DAX_CONT_EXPORT static int Run()
    {
    //create a valid openGL context that we can test transfer of data
    dax::opengl::testing::TestingWindow window;
    window.Init("Testing Window", 300, 300);

    //verify that we can transfer basic arrays and constant value arrays to opengl
    dax::testing::Testing::TryAllTypes(TransferFunctor());

    //verify that openGL interop works with all grid types in that we can
    //transfer coordinates / verts and properties to openGL
    dax::cont::testing::GridTesting::TryAllGridTypes(
                                 TransferGridFunctor(),
                                 dax::testing::Testing::CellCheckAlwaysTrue(),
                                 ArrayContainerTag(),
                                 DeviceAdapterTag() );

    return 0;
    }
};


} } }

#endif
