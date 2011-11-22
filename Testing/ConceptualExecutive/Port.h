#ifndef PORT_H
#define PORT_H

#include "DataSet.h"

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
    DataSet_(NULL), FieldType(NULL),Property(NULL)
  {}

  //copy constructor
  Port(const Port& copy_from_me):
    DataSet_(copy_from_me->DataSet_), FieldType(copy_from_me.FieldType->clone()),Property(NULL)
  {}

  //create a connection data based on the grid g, and the
  //passed in templated type
  template<typename T>
  Port(const DataSet* ds, const T&):
    DataSet_(ds), FieldType(new T),Property(NULL)
  {

  }

  template<typename T>
  Port(const Port &copy_grid_from_me, const T&):
    DataSet_(copy_grid_from_me->DataSet_), FieldType(new T),Property(NULL)
  {
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
    return 0;
  }

  std::string fieldType() const
  {
  return this->FieldType->type();
  }

  const field_type* getFieldType() const { return FieldType; }

  template<typename T>
  void setProperty(const T& t)
  {

  }

  template<typename T>
  void getProperty(T &t)
  {

  }

  bool hasProperty() const { return false;}

  bool hasModel() const { return false; }

protected:
  const field_type* FieldType;
  const DataSet* DataSet_;
};


#endif // PORT_H
