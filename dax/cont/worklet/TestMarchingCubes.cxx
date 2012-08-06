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

// --------------------------------------------------------------------includes
#include "vtkObjectFactory.h"
#include "vtkStdString.h"
#include "vtkGenericDataObjectReader.h"
#include "vtkSmartPointer.h"
#include "vtkImageData.h"
#include <vector>
#include <dax/Types.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/cont/worklet/MarchingCubes.h>
// #include <dax/cont/VectorOperations.h>

#include <dax/cont/internal/Testing.h>

class CheckValid {
public:
  CheckValid() : Valid(true) { }
  operator bool() { return this->Valid; }
  // ALWAYS INVALID FOR NOW
  void operator()(dax::Scalar value) {
      this->Valid = false;
    }
private:
  bool Valid;
};

void PrintScalarValue(dax::Scalar value)
  {
  std::cout << " " << value;
  }


template<class IteratorType>
void Print(IteratorType begin, IteratorType end)
  {
  typedef typename std::iterator_traits<IteratorType>::value_type VectorType;

  CheckValid isValid;
  for (IteratorType iter = begin; iter != end; iter++)
    {
    //VectorType vector = *iter;
    //PrintScalarValue()
    //dax::cont::VectorForEach(vector, isValid);
    //DAX_TEST_ASSERT(isValid, "Encountered bad value.");
    }
  }

template<typename T, class Container, class Device>
void Print(dax::cont::ArrayHandle<T,Container,Device> arrayHandle)
{
  Print(arrayHandle.GetPortalConstControl().GetIteratorBegin(),
              arrayHandle.GetPortalConstControl().GetIteratorEnd());
}

// ------------------------------------------------------------------------main
int main(int argc, char*argv[])
{
  float ISO_VALUE;
  vtkSmartPointer<vtkGenericDataObjectReader>
    reader = vtkSmartPointer<vtkGenericDataObjectReader>::New();

  std::cout << "filename = "
            << argv[1]
            << std::endl;

  std::cout << "Enter Iso Value : "<< std::endl;
  std::cin >> ISO_VALUE;

  reader->SetFileName(argv[1]);
  reader->Update();

  vtkImageData *volume = vtkImageData::SafeDownCast(reader->GetOutput());
  int dim[3];
  volume->GetDimensions(dim);
  int extent[6];
  std::cout << "dimensions("
            << dim[0] << ","
            << dim[1] << ","
            << dim[2] << ")"
            << std::endl;
  volume->GetExtent(extent);
  std::cout << "extent from("
            << extent[0] << ","
            << extent[2] << ","
            << extent[4] << ")"
            << std::endl;
  std::cout << "extent to("
            << extent[1] << ","
            << extent[3] << ","
            << extent[5] << ")"
            << std::endl;
  double origin[3];
  volume->GetOrigin(origin);
  std::cout << "origin ("
            << origin[0] << ","
            << origin[1] << ","
            << origin[2] << ")"
            << std::endl;

  // VTK to std::vector
  std::vector<dax::Scalar>
    field((float*)volume->GetScalarPointer(),
          (float*)volume->GetScalarPointer()+dim[0]*dim[1]*dim[2]);

  // std::vector to dax

  // Create a grid
  dax::cont::UniformGrid<> inGrid;
  inGrid.SetOrigin(dax::Vector3(origin[0], origin[1], origin[2]));
  inGrid.SetExtent(dax::make_Id3(extent[0],extent[2],extent[4]),
                   dax::make_Id3(extent[1],extent[3],extent[5]));

  // Create another triangle grid
  dax::cont::UnstructuredGrid<dax::exec::CellTriangle> outGrid;

  // Make a dax array handle to scalar field
  dax::cont::ArrayHandle<dax::Scalar> fieldHandle = dax::cont::make_ArrayHandle(field);

  //unkown size
  dax::cont::ArrayHandle<dax::Scalar> resultHandle;

  dax::Scalar isoValue = ISO_VALUE;

  try
    {
    dax::cont::worklet::MarchingCubes(inGrid,outGrid,
                                      isoValue,
                                      fieldHandle,
                                      resultHandle);
    }
  catch (dax::cont::ErrorControl error)
    {
    std::cout << "Got error: " << error.GetMessage() << std::endl;
    DAX_TEST_ASSERT(true==false,error.GetMessage());
    }

  DAX_TEST_ASSERT(resultHandle.GetNumberOfValues()==outGrid.GetNumberOfPoints(),
                  "Incorrect number of points in the result array");
  Print(resultHandle);
#if 0
  std::vector<dax::Scalar> result(resultHandle.GetNumberOfEntries());
  resultHandle.SetNewControlData(result.begin(),result.end());
  resultHandle.CompleteAsOutput(); //fetch back to control

  // FIXME: this should show up with valid results when fixed
  std::cout<< "# vtk DataFile Version 3.0" <<std::endl;
  std::cout<< "vtk output" << std::endl;
  std::cout<< "ASCII" <<std::endl;
  std::cout<< "DATASET POLYDATA" <<std::endl;
  std::cout<< "POINTS "<< (int)resultHandle.GetNumberOfEntries()/3
           << " float" <<std::endl;
  for (int i = 0; i < resultHandle.GetNumberOfEntries(); i+=3)
    {
      std::cout << result[i] << " "
                << result[i+1] << " "
                << result[i+2] << std::endl;
    }
  std::cout<<std::endl;
  std::cout << "POLYGONS "
            << (int) resultHandle.GetNumberOfEntries()/9 << " "
            << (int) resultHandle.GetNumberOfEntries()*4/9
            << std::endl;
  int j=0;
  for(int i =0 ; i < resultHandle.GetNumberOfEntries()/9;++i)
    {
      std::cout << "3 "
                << j << " "
                << j+1 << " "
                << j+2
                << std::endl;
      j+=3;
    }
#endif
  return EXIT_SUCCESS;;
}
