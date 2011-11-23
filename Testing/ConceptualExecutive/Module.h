#ifndef MODULE_H
#define MODULE_H
#include <boost/static_assert.hpp>

//------------------------------------------------------------------------------
template <typename Worklet>
struct execute
{
  template<typename A, typename B>
  void operator()(const A &input, B &output)
  {
    Worklet work;
    output.reserve(input.size());
    for(int i=0; i < input.size(); i++)
      {
      output.push_back(work.run(input[i]));
      }
  }

  template<typename A, typename B>
  void operator()(const A* input, B* output)
  {
    Worklet work;
    output->reserve(input->size());
    for(int i=0; i < input->size(); i++)
      {
      output->push_back(work.run(input->at(i)));
      }
  }
};

//class Module
//{
//public:
//  Module()
//  {}
//  virtual ~Module() {}

//protected:
//};
////------------------------------------------------------------------------------
//template < typename Worklet>
//class FieldModule : public Module//aka MapFieldModule
//{
//public:
//  FieldModule(const Port& input):
//    Module(input)
//  {
//    assert(this->outputPort().fieldType()==this->inputPort().fieldType());
//    assert(this->inputPort().hasModel()==true);
//  }

//  Worklet Work;
//};

////------------------------------------------------------------------------------
//template < typename Worklet>
//class PointToCellModule : public Module//takes a point property and converts to a cell property
//{
//public:
//  PointToCellModule(const Port& input):
//    Module(input,field_cells()) //copy inputs data, and set field to cell
//  {
//    //verify that input has a field to convert from point to cell
//    assert(this->InputPorts[0]->fieldType() == field_points().type());
//    assert(this->InputPorts[0]->hasProperty()==true);
//    assert(this->InputPorts[0]->hasModel()==true);
//  }

//  Worklet Work;
//};

////------------------------------------------------------------------------------
//template < typename Worklet>
//class CellModuleWithPointInput : public Module  //aka MapCellModule
//{
//public:
//  CellModuleWithPointInput(const Port& input):
//    Module(input,field_cells()) //copy inputs data, and set field to cell
//  {
//    //we know that the point field to use is the inputed connection data
//    //array and the inputed model is the one to operate on
//    assert(this->inputPort().fieldType()==field_points().type());
//    assert(this->inputPort().hasProperty()==true);
//    assert(this->inputPort().hasModel()==true);
//  }

//  CellModuleWithPointInput(const Port& input,
//                           const Port& pointInput):
//    Module(input,field_cells())

//  {
//    this->InputPorts.push_back(&pointInput);
//    //Use this->inputPort().Model for model
//    //Use pointInput for Data Array to process
//    assert(this->inputPort(1).fieldType()==field_points().type());
//    assert(this->inputPort(1).hasProperty()==true);
//    assert(this->inputPort(0).hasModel()==true);
//  }

//  Worklet Work;
//};

////------------------------------------------------------------------------------
//template < typename Worklet>
//class CellToPointModule: public Module  //aka Map Reduce
//{
//public:
//  CellToPointModule(const Port& input):
//    Module(input,field_points()) //copy inputs data, and set field to point
//  {
//    //verify that input has a field to convert from cell to point
//    assert(this->inputPort().fieldType() == field_cells().type());
//    assert(this->inputPort().hasProperty()==true);
//    assert(this->inputPort().hasModel()==true);
//  }

//  Worklet Work;
//};

#endif

