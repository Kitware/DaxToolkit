#ifndef MODEL_H
#define MODEL_H

#include "FilterConnecters.h"

//------------------------------------------------------------------------------
template < typename T>
class Model
{
public:
  Model(T& data):Data(&data)
  {

  }
  FilterConnector< Model<T> > pointField(const std::string& name)
  {
  return FilterConnector< Model<T> >(this,
          Port(Data, Data->pointField(name), field_points()));
  }

  FilterConnector< Model<T> > points()
  {
    return FilterConnector< Model<T> >(this,
            Port(Data, field_pointCoords()));
  }

  FilterConnector< Model<T> > cellField(const std::string& name)
  {
    return FilterConnector< Model<T> >(this,
            Port(Data, Data->cellField(name), field_cells()));
  }

  FilterConnector< Model<T> > topology()
  {
    return FilterConnector< Model<T> >(this,
            Port(Data, field_topology()));
  }

  void execute()
  {
    //maybe we can do lazy loading of the data?
    //some form of streaming maybe?

    //for now the entire data is explicit so we do nothing here
    //and it is needed for the filters pull driven model to
    //work
  }

  T* Data;
};

#endif // MODEL_H
