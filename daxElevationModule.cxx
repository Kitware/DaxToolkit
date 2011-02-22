/*=========================================================================

  This software is distributed WITHOUT ANY WARRANTY; without even
  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
  PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "daxElevationModule.h"
#include "Elevation.cl.h"

#include "daxPort.h"

daxElevationModule::daxElevationModule()
{
  this->Name = "Elevation";
  this->Code = daxHeaderString_Elevation;
  this->SetNumberOfInputs(1);
  this->SetNumberOfOutputs(1);

  daxPortPtr in_port(new daxPort());
  in_port->SetName("positions");
  this->SetInputPort(0, in_port);

  daxPortPtr out_port(new daxPort());
  out_port->SetName("output");
  this->SetOutputPort(0, out_port);
}
