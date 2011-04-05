/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxCellGradientModule.h"
#include "CellGradient.cl.h"

#include "daxPort.h"

daxCellGradientModule::daxCellGradientModule()
{
  this->Name = "CellGradient";
  this->Code = daxHeaderString_CellGradient;
  this->SetNumberOfInputs(3);
  this->SetNumberOfOutputs(1);

  daxPortPtr in_port0(new daxPort());
  in_port0->SetName("positions");
  this->SetInputPort(0, in_port0);

  daxPortPtr in_port1(new daxPort());
  in_port1->SetName("connections");
  this->SetInputPort(1, in_port1);

  daxPortPtr in_port2(new daxPort());
  in_port2->SetName("input_point");
  this->SetInputPort(2, in_port2);

  daxPortPtr out_port(new daxPort());
  out_port->SetName("output_cell");
  this->SetOutputPort(0, out_port);
}
