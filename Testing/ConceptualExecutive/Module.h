#ifndef MODULE_H
#define MODULE_H

#include "Port.h"

#include <vector>
#include <assert.h>

//------------------------------------------------------------------------------
template < typename T>
class Model
{
public:
  Model(T& data):Data(data)
  {

  }
  Port pointField() const
  {
    return Port(Data,field_points());
  }
  Port cellField() const
  {
    return Port(Data,field_cells());
  }
  T Data;
};

//------------------------------------------------------------------------------
class Module
{
public:
  Module(const Port& input):
    OutputPort(input),
    Mergeable(input.hasProperty())
  {
    this->InputPorts.push_back(&input);
  }

  Module(const Port& input, const field_points& field):
    OutputPort(input,field),
    Mergeable(input.fieldType()==field.type())
    {
    this->InputPorts.push_back(&input);
    }

  Module(const Port& input, const field_cells& field):
    OutputPort(input,field),
    Mergeable(input.fieldType()==field.type())
    {
    this->InputPorts.push_back(&input);
    }

  virtual ~Module() {}
  virtual const Port& outputPort() const
  {
    return OutputPort;
  }

  virtual const Port& inputPort(const int idx=0) const
  {
    return *(InputPorts[idx]);
  }

  virtual size_t numberOfInputPorts() const { return InputPorts.size(); }

  virtual size_t numberOfOutputPorts()  const { return 1; }

  bool isMergeable() const { return Mergeable; }

protected:
  bool Mergeable;
  std::vector<const Port*> InputPorts;
  Port OutputPort;
};

//------------------------------------------------------------------------------
template < typename T>
class FieldModule : public Module//aka MapFieldModule
{
public:
  FieldModule(const Port& input):
    Module(input)
  {
    assert(this->outputPort().fieldType()==this->inputPort().fieldType());
    assert(this->inputPort().hasModel()==true);

    this->OutputPort.setProperty(Result);
    assert(this->outputPort().hasProperty()==true);
  }

  std::vector<T> Result;
};

//------------------------------------------------------------------------------
template < typename T>
class PointToCellModule : public Module//takes a point property and converts to a cell property
{
public:
  PointToCellModule(const Port& input):
    Module(input,field_cells()) //copy inputs data, and set field to cell
  {
    //verify that input has a field to convert from point to cell
    assert(this->InputPorts[0]->fieldType() == field_points().type());
    assert(this->InputPorts[0]->hasProperty()==true);
    assert(this->InputPorts[0]->hasModel()==true);
    this->OutputPort.setProperty(Result);
  }

  std::vector<T> Result;
};

//------------------------------------------------------------------------------
template < typename T>
class CellModuleWithPointInput : public Module  //aka MapCellModule
{
public:
  CellModuleWithPointInput(const Port& input):
    Module(input,field_cells()) //copy inputs data, and set field to cell
  {
    //we know that the point field to use is the inputed connection data
    //array and the inputed model is the one to operate on
    assert(this->inputPort().fieldType()==field_points().type());
    assert(this->inputPort().hasProperty()==true);
    assert(this->inputPort().hasModel()==true);
    this->OutputPort.setProperty(Result);
  }

  CellModuleWithPointInput(const Port& input,
                           const Port& pointInput):
    Module(input,field_cells())

  {
    this->InputPorts.push_back(&pointInput);
    //Use this->inputPort().Model for model
    //Use pointInput for Data Array to process
    assert(this->inputPort(1).fieldType()==field_points().type());
    assert(this->inputPort(1).hasProperty()==true);
    assert(this->inputPort(0).hasModel()==true);
   this->OutputPort.setProperty(Result);
  }

  std::vector<T> Result;
};

//------------------------------------------------------------------------------
template < typename T>
class CellToPointModule: public Module  //aka Map Reduce
{
public:
  CellToPointModule(const Port& input):
    Module(input,field_points()) //copy inputs data, and set field to point
  {
    //verify that input has a field to convert from cell to point
    assert(this->inputPort().fieldType() == field_cells().type());
    assert(this->inputPort().hasProperty()==true);
    assert(this->inputPort().hasModel()==true);
    this->OutputPort.setProperty(Result);
  }

  std::vector<T> Result;
};

#endif

