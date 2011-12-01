#ifndef CONCEPT_H
#define CONCEPT_H

#include "StaticAssert.h"
#include <dax/Types.h>

#include <vector>
#include <assert.h>

#include <boost/shared_ptr.hpp>
#include <dax/cont/StructuredGrid.h>


//------------------------------------------------------------------------------
struct field_type
{
  field_type(){}

  virtual ~field_type()
  {
  }
  virtual int size(const dax::cont::DataSet* grid) const = 0;
  virtual std::string type() const = 0;
  virtual field_type* clone() const = 0;  
  std::string name;
};

//------------------------------------------------------------------------------
struct field_unkown : public field_type
{

  field_unkown()
  {
    name="unkown";
  }
  ~field_unkown(){}

  int size(const dax::cont::DataSet* grid) const
  {
    return 0;
  }

  std::string type() const {return name;}

  field_unkown* clone() const
  {
    return new field_unkown;
  }
};

//------------------------------------------------------------------------------
struct field_points : public field_type
{

  field_points()
  {
    name="points";
  }
  ~field_points(){}

  int size(const dax::cont::DataSet* grid) const
  {
    return grid->numPoints();
  }

  std::string type() const {return name;}

  field_points* clone() const
  {
    return new field_points;
  }
};

//------------------------------------------------------------------------------
struct field_cells : public field_type
{
  field_cells()
  {
    name="cells";
  }

  ~field_cells(){}

  int size(const dax::cont::DataSet* grid) const
  {
    return grid->numCells();
  }

  std::string type() const {return name;}

  field_cells* clone() const
  {
    return new field_cells;
  }
};

//------------------------------------------------------------------------------
class Port
{
public:

  //default constructor with an empty DataSet_
  Port():
    DataSet_(NULL), FieldType(new field_unkown())
  {
  }

  Port(Port& copy_from_me):
    DataSet_(copy_from_me.DataSet_), FieldType(copy_from_me.FieldType->clone())
  {
  }

  //copy constructor
  Port(const Port& copy_from_me):
    DataSet_(copy_from_me.DataSet_), FieldType(copy_from_me.FieldType->clone())
  {
  }

  //create a connection data based on the DataSet_ g, and the
  //passed in templated type
  template<typename T>
  Port(dax::cont::DataSet *g, const T& t):
    DataSet_(g), FieldType(t.clone())
  {
  }

  template<typename T>
  Port(const Port &copy_DataSet__from_me, const T& t):
    DataSet_(copy_DataSet__from_me.DataSet_), FieldType(t.clone())
  {    
  }

  Port& operator=(const Port& op)
  {
    this->DataSet_=op.DataSet_;

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
    return this->FieldType->size(this->DataSet_);
  }

  std::string fieldType() const
  {
  return this->FieldType->type();
  }

  const field_type& getFieldType() const { return *FieldType; }

  bool hasModel() const { return (this->DataSet_ &&
                                  this->DataSet_->numCells() > 0 &&
                                  this->DataSet_->numPoints() > 0); }


  bool isValid() const { return FieldType!=NULL; }

  void initProp() const
  {
    //hack to make sure the FauxProperty is the right size
    this->FauxProperty.resize(this->FieldType->size(this->DataSet_),1);
  }

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

  dax::cont::DataSet* dataSet() const
  {
    return this->DataSet_;
  }

protected:
  std::vector<dax::Scalar> FauxProperty;
  field_type* FieldType;
  dax::cont::DataSet* DataSet_;
};

//------------------------------------------------------------------------------
template < typename T>
class Model
{
public:
  Model(T& data):Data(&data)
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

  T* Data;
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
    return *(&InputPorts[idx]);
  }

  virtual size_t numberOfInputPorts() const { return InputPorts.size(); }

  virtual size_t numberOfOutputPorts()  const { return OutputPorts.size(); }

protected:
  std::vector<Port> InputPorts;
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
    this->InputPorts.push_back(input1);
    this->InputPorts.push_back(input2);
    this->InputPorts.push_back(input3);
    this->InputPorts.push_back(input4);
    this->InputPorts.push_back(input5);
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
template < typename Worklet>
class ChangeDataSetModule: public Module  //aka Change Topology
{
public:
   //copy inputs data, and set field to point
  ChangeDataSetModule(const Port& input):
    Module(Worklet::NumInputs,Worklet::NumOutputs,input)
  {
    STATIC_ASSERT(Worklet::NumInputs==1,Incorrect_Number_Of_Parameters);

    //instead of using the default output ports we need to set
    //them to the new data set that this worklet is generating
    std::vector<Port>::iterator it;
    Worklet work;
    for(int i=0;i<this->numberOfOutputPorts();++i)
      {
      this->OutputPorts[i] = Port(work.requestDataSet(i,this->InputPorts),
                                  this->InputPorts[i].getFieldType());
      }

  }

  void run()
  {
    std::cout << "Executing " << Worklet().name() << std::endl;
    Worklet()(this->InputPorts,this->OutputPorts);
  }
};

//------------------------------------------------------------------------------
template < typename ModuleFunction >
class Filter
{
public:

  Filter(const Port& input):
    Module_(input)
  {}

  Filter(const Port& input, const Port& input2):
    Module_(input,input2)
  {}

  Filter(const Port& input, const Port& input2, const Port& input3):
    Module_(input,input2,input3)
  {}

  Filter(const Port& input, const Port& input2, const Port& input3,
         const Port& input4):
    Module_(input,input2,input3,input4)
  {}


  template <typename ModuleFunctionA>
  Filter(Filter<ModuleFunctionA>& inputFilter):
    Module_(inputFilter.outputPort())
  {

  }

  template <typename ModuleFunctionA, typename ModuleFunctionB>
  Filter(Filter<ModuleFunctionA>& inputFilter, Filter<ModuleFunctionB>& inputFilter2):
    Module_(inputFilter.outputPort(),
          inputFilter2.outputPort())
  {

  }

  template <typename ModuleFunctionA, typename ModuleFunctionB, typename ModuleFunctionC>
  Filter(Filter<ModuleFunctionA>& inputFilterA, Filter<ModuleFunctionB>& inputFilterB,
         Filter<ModuleFunctionC>& inputFilterC):
    Module_(inputFilterA.outputPort(),
          inputFilterB.outputPort(),
          inputFilterC.outputPort())
  {

  }

  template <typename ModuleFunctionA, typename ModuleFunctionB, typename ModuleFunctionC, typename ModuleFunctionD>
  Filter(Filter<ModuleFunctionA>& inputFilterA, Filter<ModuleFunctionB>& inputFilterB,
          Filter<ModuleFunctionC>& inputFilterC, Filter<ModuleFunctionD>& inputFilterD):
    Module_(inputFilterA.outputPort(),
          inputFilterB.outputPort(),
          inputFilterC.outputPort(),
          inputFilterD.outputPort())
  {

  }

  const Port& outputPort(const int index=0) const
  {
    return Module_.outputPort(index);
  }

  std::string fieldType(const int index=0) const
  {
    return this->outputPort(index).fieldType();
  }

  int size(const int index=0) const
  {
    return this->outputPort(index).size();
  }

  void run()
  {
    std::cout << "starting run" << std::endl;
     Module_.run();
  }

  ModuleFunction Module_;
};

#endif

