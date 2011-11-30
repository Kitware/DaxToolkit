#ifndef CONCEPT_H
#define CONCEPT_H

#include "StaticAssert.h"
#include <dax/Types.h>

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

  int NumPoints;
  int NumCells;
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
    G(Grid(-1,-1)), FieldType(NULL)
  {}

  //copy constructor
  Port(const Port& copy_from_me):
    G(copy_from_me.G), FieldType(copy_from_me.FieldType->clone())
  {
    //hack to make sure the FauxProperty is the right size
    this->FauxProperty.resize(this->FieldType->size(this->G),1);
  }

  //create a connection data based on the grid g, and the
  //passed in templated type
  template<typename T>
  Port(const Grid &g, const T&):
    G(g), FieldType(new T)
  {
    //hack to make sure the FauxProperty is the right size
    this->FauxProperty.resize(this->FieldType->size(this->G),1);
  }

  template<typename T>
  Port(const Port &copy_grid_from_me, const T&):
    G(copy_grid_from_me.G), FieldType(new T)
  {
    //hack to make sure the FauxProperty is the right size
    this->FauxProperty.resize(this->FieldType->size(this->G),1);
  }

  Port& operator=(const Port& op)
  {
    this->G=op.G;

    if(this->FieldType)
      {
      delete this->FieldType;
      }

    this->FieldType = op.FieldType->clone();

    return *this;
  }

  ~Port()
  {
    if(FieldType)
      {
      delete FieldType;
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

  bool hasModel() const { return (G.NumCells > 0 && G.NumPoints > 0); }


  bool isValid() const { return FieldType!=NULL; }

  //the port should be holding an array that
  //we than using an iterator on!
  dax::Scalar at(int idx) const
  {
    return FauxProperty[idx];
  }

  dax::Scalar& set(int idx)
  {
    return *(&FauxProperty[idx]);
  }

protected:
  std::vector<dax::Scalar> FauxProperty;
  field_type* FieldType;
  Grid G;
};

//------------------------------------------------------------------------------
template < typename T>
class Model
{
public:
  Model(T& data):Data(data)
  {

  }
  Port pointField(const std::string& name="") const
  {
    return Port(Data,field_points());
  }

  Port points() const
  {
    return Port(Data,field_points());
  }

  Port cellField(const std::string& name="") const
  {
    return Port(Data,field_cells());
  }

  Port cells() const
  {
    return Port(Data,field_cells());
  }

  T Data;
};

//------------------------------------------------------------------------------
class Module
{
public:
  Module(const int &numInputPorts, const int &numOutputPorts,
         const Port& input1,
         const Port& input2 = Port(),
         const Port& input3 = Port(),
         const Port& input4 = Port(),
         const Port& input5 = Port()):
    NumInputs(numInputPorts),
    NumOutputs(numOutputPorts)
  {
    this->init(input1,input1,input2,input3,input4,input5);
  }

  Module(const int& numInputPorts, const int &numOutputPorts,
         const field_points& field,
         const Port& input1,
         const Port& input2 = Port(),
         const Port& input3 = Port(),
         const Port& input4 = Port(),
         const Port& input5 = Port()):
    NumInputs(numInputPorts),
    NumOutputs(numOutputPorts)
    {
    //create our output port as using the same data
    //as the input port but with the new field
    //copy input port as our ouput port
    this->init(Port(input1,field),input1,input2,input3,input4,input5);
    }

  Module(const int& numInputPorts, const int &numOutputPorts,
         const field_cells& field,
         const Port& input1,
         const Port& input2 = Port(),
         const Port& input3 = Port(),
         const Port& input4 = Port(),
         const Port& input5 = Port()):
    NumInputs(numInputPorts),
    NumOutputs(numOutputPorts)
    {
    //create our output port as using the same data
    //as the input port but with the new field
    this->init(Port(input1,field),input1,input2,input3,input4,input5);
    }

  virtual ~Module() {}

  virtual const Port& outputPort( const int idx=0 ) const
  {
    return *(&OutputPorts[idx]);
  }

  virtual Port& outputPort( const int idx=0 )
  {
    return *(&OutputPorts[idx]);
  }

  virtual const Port& inputPort(const int idx=0) const
  {
    return *(InputPorts[idx]);
  }

  virtual size_t numberOfInputPorts() const { return InputPorts.size(); }

  virtual size_t numberOfOutputPorts()  const { return OutputPorts.size(); }

protected:
  std::vector<const Port*> InputPorts;
  std::vector<Port> OutputPorts;
  const int NumInputs;
  const int NumOutputs;

private:
  void init(Port output, const Port& input1, const Port& input2,
            const Port& input3, const Port& input4, const Port& input5)
  {
    //copy input port as our ouput port
    this->OutputPorts = std::vector<Port>(NumOutputs,output);

    //keep a reference to all the used input ports
    this->InputPorts.push_back(&input1);
    this->InputPorts.push_back(&input2);
    this->InputPorts.push_back(&input3);
    this->InputPorts.push_back(&input4);
    this->InputPorts.push_back(&input5);
    this->InputPorts.resize(NumInputs);
  }

};

//------------------------------------------------------------------------------
template < typename Worklet>
class FieldModule : public Module//aka MapFieldModule
{
public:
  FieldModule(const Port& input):
    Module(Worklet::NumInputs,Worklet::NumOutputs,input)
  {
    STATIC_ASSERT(Worklet::NumInputs==1,Incorrect_Number_Of_Parameters);
  }

  FieldModule(const Port& input1, const Port& input2):
    Module(Worklet::NumInputs,Worklet::NumOutputs,input1,input2)
  {
    STATIC_ASSERT(Worklet::NumInputs==2,Incorrect_Number_Of_Parameters);
  }

  FieldModule(const Port& input1, const Port& input2, const Port& input3):
    Module(Worklet::NumInputs,Worklet::NumOutputs,input1,input2, input3)
  {
    STATIC_ASSERT(Worklet::NumInputs==3,Incorrect_Number_Of_Parameters);
  }

  FieldModule(const Port& input1, const Port& input2,
              const Port& input3, const Port& input4):
    Module(Worklet::NumInputs,Worklet::NumOutputs,input1,input2, input3, input4)
  {
    STATIC_ASSERT(Worklet::NumInputs==4,Incorrect_Number_Of_Parameters);
  }

  void run()
  {
    std::cout << "Executing " << Worklet().name() << std::endl;
    Worklet()(this->InputPorts,this->OutputPorts);
  }

};

//------------------------------------------------------------------------------
template < typename Worklet>
class PointToCellModule : public Module//takes a point property and converts to a cell property
{
public:
  //copy inputs data, and set field to cell
  PointToCellModule(const Port& input):
    Module(Worklet::NumInputs,Worklet::NumOutputs,field_cells(),input)
  {    
    STATIC_ASSERT(Worklet::NumInputs==1,Incorrect_Number_Of_Parameters);
  }

  void run()
  {
    std::cout << "Executing " << Worklet().name() << std::endl;
    Worklet()(this->InputPorts,this->OutputPorts);
  }
};

//------------------------------------------------------------------------------
template < typename Worklet>
class CellModuleWithPointInput : public Module  //aka MapCellModule
{
public:  
  CellModuleWithPointInput(const Port& input,const Port& pointInput):
    Module(Worklet::NumInputs,Worklet::NumOutputs,field_cells(),input,pointInput)

  {   
    STATIC_ASSERT(Worklet::NumInputs==2,Incorrect_Number_Of_Parameters);
  }

  void run()
  {
    std::cout << "Executing " << Worklet().name() << std::endl;
    Worklet()(this->InputPorts,this->OutputPorts);
  }
};

//------------------------------------------------------------------------------
template < typename Worklet>
class CellToPointModule: public Module  //aka Map Reduce
{
public:
   //copy inputs data, and set field to point
  CellToPointModule(const Port& input):
    Module(Worklet::NumInputs,Worklet::NumOutputs,field_points(),input)
  {
    STATIC_ASSERT(Worklet::NumInputs==1,Incorrect_Number_Of_Parameters);
  }

  void run()
  {
    std::cout << "Executing " << Worklet().name() << std::endl;
    Worklet()(this->InputPorts,this->OutputPorts);
  }
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
    Funct(inputFilter.outputPort())
  {

  }

  template <typename FunctorA, typename FunctorB>
  Filter(Filter<FunctorA>& inputFilter, Filter<FunctorB>& inputFilter2):
    Funct(inputFilter.outputPort(),
          inputFilter2.outputPort())
  {

  }

  template <typename FunctorA, typename FunctorB, typename FunctorC>
  Filter(Filter<FunctorA>& inputFilterA, Filter<FunctorB>& inputFilterB,
         Filter<FunctorC>& inputFilterC):
    Funct(inputFilterA.outputPort(),
          inputFilterB.outputPort(),
          inputFilterC.outputPort())
  {

  }

  template <typename FunctorA, typename FunctorB, typename FunctorC, typename FunctorD>
  Filter(Filter<FunctorA>& inputFilterA, Filter<FunctorB>& inputFilterB,
          Filter<FunctorC>& inputFilterC, Filter<FunctorD>& inputFilterD):
    Funct(inputFilterA.outputPort(),
          inputFilterB.outputPort(),
          inputFilterC.outputPort(),
          inputFilterD.outputPort())
  {

  }

  const Port& outputPort(const int index=0) const
  {
    return Funct.outputPort(index);
  }

  std::string fieldType(const int index=0) const
  {
    return this->outputPort(index).fieldType();
  }

  std::string isMergeable() const
  {
    return Funct.isMergeable() ? "True":"False";
  }

  void run()
  {
    Funct.run();
  }

  Function Funct;
};

#endif

