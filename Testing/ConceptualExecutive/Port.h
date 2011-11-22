#ifndef PORT_H
#define PORT_H

#include "HeterogeneousContainer.h"

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


#endif // PORT_H
