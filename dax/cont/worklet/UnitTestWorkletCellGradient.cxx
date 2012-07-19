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

#include <dax/cont/ArrayContainerControlBasic.h>
#include <dax/cont/DeviceAdapterSerial.h>

// These header files help tease out when the default template arguments to
// ArrayHandle are inappropriately used.
#include <dax/cont/internal/ArrayContainerControlError.h>
#include <dax/cont/internal/DeviceAdapterError.h>

#include <dax/cont/worklet/CellGradient.h>
#include <dax/VectorTraits.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>

#include <dax/cont/internal/Testing.h>

#include <vector>

namespace {

const dax::Id DIM = 64;

//-----------------------------------------------------------------------------
static void TestCellGradient()
{
  dax::cont::UniformGrid<dax::cont::ArrayContainerControlTagBasic,
                         dax::cont::DeviceAdapterTagSerial> grid;
  grid.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(DIM-1, DIM-1, DIM-1));

  dax::Vector3 trueGradient = dax::make_Vector3(1.0, 1.0, 1.0);

  std::vector<dax::Scalar> field(grid.GetNumberOfPoints());
  for (dax::Id pointIndex = 0;
       pointIndex < grid.GetNumberOfPoints();
       pointIndex++)
    {
    field[pointIndex]
        = dax::dot(grid.ComputePointCoordinates(pointIndex), trueGradient);
    }
  dax::cont::ArrayHandle<dax::Scalar,
                        dax::cont::ArrayContainerControlTagBasic,
                        dax::cont::DeviceAdapterTagSerial> fieldHandle =
      dax::cont::make_ArrayHandle(field,
                                  dax::cont::ArrayContainerControlTagBasic(),
                                  dax::cont::DeviceAdapterTagSerial());

  dax::cont::ArrayHandle<dax::Vector3,
                         dax::cont::ArrayContainerControlTagBasic,
                         dax::cont::DeviceAdapterTagSerial> gradientHandle;

  std::cout << "Running CellGradient worklet" << std::endl;
  dax::cont::worklet::CellGradient(grid,
                                   grid.GetPointCoordinates(),
                                   fieldHandle,
                                   gradientHandle);

  std::cout << "Checking result" << std::endl;
  std::vector<dax::Vector3> gradient(grid.GetNumberOfCells());
  gradientHandle.CopyInto(gradient.begin());
  for (dax::Id cellIndex = 0;
       cellIndex < grid.GetNumberOfCells();
       cellIndex++)
    {
    dax::Vector3 gradientValue = gradient[cellIndex];
    DAX_TEST_ASSERT(test_equal(gradientValue, trueGradient),
                    "Got bad gradient");
    }
}

} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletCellGradient(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestCellGradient);
}
