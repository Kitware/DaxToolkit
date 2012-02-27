/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

===========================================================================*/
#include <stdio.h>
#include <iostream>
#include "Timer.h"

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/cont/VectorOperations.h>

#include <dax/cont/worklet/Elevation.h>
#include <dax/cont/worklet/Threshold.h>

#include <vector>

#include <vtkNew.h>
#include <vtkFloatArray.h>
#include <vtkImageData.h>
#include <vtkThreshold.h>
#include <vtkPointData.h>
#include <vtkSmartPointer.h>
#include <vtkUnstructuredGrid.h>


/*
#include "piston/image3d.h"
#include "piston/threshold_geometry.h"
#include "piston/vtk_image3d.h"


using namespace piston;
#define SPACE thrust::detail::default_device_space_tag
*/


namespace
{
void PrintResults(int pipeline, double time, const char* name)
{
  std::cout << "Elapsed time: " << time << " seconds." << std::endl;
  std::cout << "CSV, " << name <<" ,"
            << pipeline << "," << time << std::endl;
}

void RunVTKPipeline(const dax::cont::UniformGrid &dgrid, vtkImageData* grid)
{
  std::cout << "Running pipeline 1: Elevation -> Threshold" << std::endl;

  std::vector<dax::Scalar> elev(dgrid.GetNumberOfPoints());
  dax::cont::ArrayHandle<dax::Scalar> elevHandle(elev.begin(),elev.end());

  //use dax to compute the elevation
  dax::cont::worklet::Elevation(dgrid, dgrid.GetPoints(), elevHandle);

  //now that the results are back on the cpu, do threshold with VTK
  vtkSmartPointer<vtkFloatArray> vtkElevationPoints = vtkSmartPointer<vtkFloatArray>::New();
  vtkElevationPoints->SetName("Elevation");
  vtkElevationPoints->SetVoidArray(&elev[0],elev.size(),1);
  grid->GetPointData()->AddArray(vtkElevationPoints);

  vtkNew<vtkThreshold> threshold;
  threshold->SetInput(grid);
  threshold->AllScalarsOn();
  threshold->ThresholdBetween(0, 100);
  threshold->SetInputArrayToProcess(0, 0, 0, vtkDataObject::FIELD_ASSOCIATION_POINTS,"Elevation");

  Timer timer;
  threshold->Update();
  double time = timer.elapsed();

  vtkSmartPointer<vtkUnstructuredGrid> out = threshold->GetOutput();
  
  std::cout << "original GetNumberOfCells: " << dgrid.GetNumberOfCells() << std::endl;
  std::cout << "threshold GetNumberOfCells: " << out->GetNumberOfCells() << std::endl;
  PrintResults(1, time, "VTK");
}

/*
void RunPISTONPipeline(const dax::cont::UniformGrid &dgrid, vtkImageData* grid)
{
  std::cout << "Running pipeline 1: Elevation -> Threshold" << std::endl;

  std::vector<dax::Scalar> elev(dgrid.GetNumberOfPoints());
  dax::cont::ArrayHandle<dax::Scalar> elevHandle(elev.begin(),elev.end());

  //use dax to compute the elevation
  dax::cont::worklet::Elevation(dgrid, dgrid.GetPoints(), elevHandle);

  //now that the results are back on the cpu, do threshold with VTK
  vtkSmartPointer<vtkFloatArray> vtkElevationPoints = vtkSmartPointer<vtkFloatArray>::New();
  vtkElevationPoints->SetName("Elevation");
  vtkElevationPoints->SetVoidArray(&elev[0],elev.size(),1);

  grid->GetPointData()->SetScalars(vtkElevationPoints); //piston on works on active scalars
  vtk_image3d<int,float,SPACE> image =  vtk_image3d<int, float, SPACE>(grid);
  threshold_geometry<vtk_image3d<int, float, SPACE> > threshold(image,0,100);
  threshold.set_threshold_range(0,100);

  Timer timer;
  threshold();
  double time = timer.elapsed();



  std::cout << "original GetNumberOfCells: " << dgrid.GetNumberOfCells() << std::endl;
  std::cout << "threshold GetNumberOfCells: " << threshold.valid_cell_indices.size() << std::endl;
  PrintResults(1, time, "Piston");
}
*/



} // Anonymous namespace

