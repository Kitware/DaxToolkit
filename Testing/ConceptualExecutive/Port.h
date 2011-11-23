#ifndef PORT_H
#define PORT_H

//------------------------------------------------------------------------------
struct field_type
{
  virtual std::string type() const {return "field";}
  virtual field_type* clone() const = 0;
};

//------------------------------------------------------------------------------
struct field_points : public field_type
{

  std::string type() const {return "points";}

  field_points* clone() const
  {
    return new field_points;
  }
};

//------------------------------------------------------------------------------
struct field_cells : public field_type
{
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
    FieldType(NULL)
  {}

  //copy constructor
  Port(const Port& copy_from_me):
    FieldType(copy_from_me.FieldType->clone())
  {}

  //create a connection data based on the grid g, and the
  //passed in templated type
  template<typename D, typename T>
  Port(const D* ds, const T&):
    FieldType(new T)
  {

  }

  template<typename T>
  Port(const Port &copy_grid_from_me, const T&):
    FieldType(new T)
  {
  }

  ~Port()
  {
    if(FieldType)
      {
      delete FieldType;
      }
  }

  std::string fieldType() const
  {
  return this->FieldType->type();
  }

  const field_type* getFieldType() const { return FieldType; }


protected:
  const field_type* FieldType;
};


#endif // PORT_H
