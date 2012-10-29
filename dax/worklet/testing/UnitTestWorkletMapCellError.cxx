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

#include <dax/worklet/testing/CellMapError.worklet>

#include <dax/VectorTraits.h>

#include <dax/cont/ArrayContainerControl.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/ErrorExecution.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/UniformGrid.h>

#include <dax/cont/internal/testing/Testing.h>

namespace {

const dax::Id DIM = 64;

//-----------------------------------------------------------------------------
static void TestCellMapError()
{
  dax::cont::UniformGrid<> grid;
  grid.SetExtent(dax::make_Id3(0, 0, 0), dax::make_Id3(DIM-1, DIM-1, DIM-1));

  std::cout << "Running field map worklet that errors" << std::endl;
  bool gotError = false;
  try
    {
    dax::cont::Scheduler<> scheduler;
    scheduler.Invoke(dax::worklet::testing::CellMapError(),grid);
    }
  catch (dax::cont::ErrorExecution error)
    {
    std::cout << "Got expected ErrorExecution object." << std::endl;
    std::cout << error.GetMessage() << std::endl;
    gotError = true;
    }

  DAX_TEST_ASSERT(gotError, "Never got the error thrown.");
}

} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletMapCellError(int, char *[])
{
  return dax::cont::internal::Testing::Run(TestCellMapError);
}
