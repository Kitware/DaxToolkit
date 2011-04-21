/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxCellDataToPointDataModule.h"
#include "CellDataToPointData.cl.h"

#include "daxPort.h"

daxCellDataToPointDataModule::daxCellDataToPointDataModule()
{
  this->Name = "CellDataToPointData";
  this->Code = daxHeaderString_CellDataToPointData;
  this->SetNumberOfInputs(2);
  this->SetNumberOfOutputs(1);

  daxPortPtr in_port0(new daxPort());
  in_port0->SetName("cell_links");
  this->SetInputPort(0, in_port0);

  daxPortPtr in_port1(new daxPort());
  in_port1->SetName("in_cell_array");
  this->SetInputPort(1, in_port1);

  daxPortPtr in_port2(new daxPort());
  in_port2->SetName("out_point_array");
  this->SetOutputPort(0, in_port2);
}
