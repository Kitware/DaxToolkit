#ifndef RUNTIMETRAITS_H
#define RUNTIMETRAITS_H

#include "HeterogeneousContainer.h"

#include <vector>
#include <assert.h>

//------------------------------------------------------------------------------
struct Grid
{
  //copy constructor
  Grid():
    NumPoints(-1),NumCells(-1)
  {}

  Grid(const Grid& copy_from_me):
    NumPoints(copy_from_me.NumPoints),NumCells(copy_from_me.NumCells)
  {}

  Grid(const int Points, const int Cells):NumPoints(Points),NumCells(Cells)
  {}

  const int NumPoints;
  const int NumCells;
};

//------------------------------------------------------------------------------
int numberOfPoints(const Grid &gridstructure)
{
  return gridstructure.NumPoints;
}

//------------------------------------------------------------------------------
int numberOfCells(const Grid &gridstructure)
{
  return gridstructure.NumCells;
}

//------------------------------------------------------------------------------
struct field_type
{
  virtual int size(const Grid &grid) const {return -1;}
  virtual std::string type() const {return "field";}
  virtual field_type* clone() const = 0;
};

//------------------------------------------------------------------------------
struct field_points : public field_type
{

  int size(const Grid &grid) const
  {
    return numberOfPoints(grid);
  }

  std::string type() const {return "points";}

  field_points* clone() const
  {
    return new field_points;
  }
};

//------------------------------------------------------------------------------
struct field_cells : public field_type
{
  int size(const Grid &grid) const
  {
    return numberOfCells(grid);
  }

  std::string type() const {return "cells";}

  field_cells* clone() const
  {
    return new field_cells;
  }
};

//------------------------------------------------------------------------------
class Port
{
public:

  //default constructor with an empty grid
  Port():
    G(Grid(-1,-1)), FieldType(NULL),Property(NULL)
  {}

  //copy constructor
  Port(const Port& copy_from_me):
    G(copy_from_me.G), FieldType(copy_from_me.FieldType->clone()),Property(NULL)
  {}

  //create a connection data based on the grid g, and the
  //passed in templated type
  template<typename T>
  Port(const Grid &g, const T&):
    G(g), FieldType(new T),Property(NULL)
  {

  }

  template<typename T>
  Port(const Port &copy_grid_from_me, const T&):
    G(copy_grid_from_me.G), FieldType(new T),Property(NULL)
  {
  }

  ~Port()
  {
    if(FieldType)
      {
      delete FieldType;
      }
    if(Property)
      {
      delete Property;
      }
  }

  int size() const
  {
    return this->FieldType->size(this->G);
  }

  std::string fieldType() const
  {
  return this->FieldType->type();
  }

  const field_type* getFieldType() const { return FieldType; }

  template<typename T>
  void setProperty(const T& t)
  {
    Property = new ObjectHandleT<T>(t);
  }

  template<typename T>
  void getProperty(T &t)
  {
    t = static_cast<ObjectHandleT<T>*>(Property)->get();
  }

  bool hasProperty() const { return Property!=NULL;}

  bool hasModel() const { return (G.NumCells > 0 && G.NumPoints > 0); }

protected:
  const field_type* FieldType;
  const Grid G;
  ObjectHandle* Property;
};

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

//------------------------------------------------------------------------------
template < typename Functor >
class Filter
{
public:
  typedef Functor Function;

  Filter(const Port& input):
    Funct(input)
  {}

  Filter(const Port& input, const Port& input2):
    Funct(input,input2)
  {}

  Filter(const Port& input, const Port& input2, const Port& input3):
    Funct(input,input2,input3)
  {}

  Filter(const Port& input, const Port& input2, const Port& input3,
         const Port& input4):
    Funct(input,input2,input3,input4)
  {}


  template <typename FunctorA>
  Filter(Filter<FunctorA>& inputFilter):
    Funct(inputFilter.Funct.outputPort())
  {

  }

  template <typename FunctorA, typename FunctorB>
  Filter(Filter<FunctorA>& inputFilter, Filter<FunctorB>& inputFilter2):
    Funct(inputFilter.Funct.outputPort(),
          inputFilter2.Funct.outputPort())
  {

  }

  template <typename FunctorA, typename FunctorB, typename FunctorC>
  Filter(Filter<FunctorA>& inputFilterA, Filter<FunctorB>& inputFilterB,
         Filter<FunctorC>& inputFilterC):
    Funct(inputFilterA.Funct.outputPort(),
          inputFilterB.Funct.outputPort(),
          inputFilterC.Funct.outputPort())
  {

  }

  template <typename FunctorA, typename FunctorB, typename FunctorC, typename FunctorD>
  Filter(Filter<FunctorA>& inputFilterA, Filter<FunctorB>& inputFilterB,
          Filter<FunctorC>& inputFilterC, Filter<FunctorD>& inputFilterD):
    Funct(inputFilterA.Funct.outputPort(),
          inputFilterB.Funct.outputPort(),
          inputFilterC.Funct.outputPort(),
          inputFilterD.Funct.outputPort())
  {

  }

  const Port& OutputPort( ) const
  {
    return Funct.outputPort();
  }

  int size()
  {
    return Funct.outputPort().size();
  }

  std::string fieldType() const
  {
    return Funct.outputPort().fieldType();
  }

  std::string isMergeable() const
  {
    return Funct.isMergeable() ? "True":"False";
  }

  Function Funct;
};

#endif

