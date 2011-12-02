#ifndef PORTGENERATORS_H
#define PORTGENERATORS_H

#include "Concept.h"
#include <dax/cont/DataSet.h>

//------------------------------------------------------------------------------
// Functions that generate Port objects, a potential way around
// having models and making it easier to make filter connections
Port pointField(dax::cont::DataSet& Data, const std::string& name)
{
  return Port(&Data, Data.pointField(name), field_points());
}

Port pointField(const Port &port, const std::string& name)
{
  return Port(port.dataSet(), port.dataSet()->pointField(name), field_points());
}

template <typename T>
Port pointField(const Filter<T>& filter, const std::string& name)
{
  return pointField(filter.outputPort(),name);
}

//------------------------------------------------------------------------------
Port cellField(dax::cont::DataSet& Data, const std::string& name )
{
  return Port(&Data, Data.cellField(name), field_cells());
}

Port cellField(const Port &port, const std::string& name )
{
  return Port(port.dataSet(), port.dataSet()->cellField(name), field_cells());
}

template <typename T>
Port cellField(const Filter<T>& filter, const std::string& name )
{
  return cellField(filter.outputPort(),name);
}

//------------------------------------------------------------------------------
Port points(dax::cont::DataSet& Data)
{
  return Port(&Data, field_pointCoords());
}

Port points(const Port &port)
{
  return Port(port.dataSet(), field_pointCoords());
}

template <typename T>
Port points(const Filter<T>& filter)
{
  return points(filter.outputPort());
}

//------------------------------------------------------------------------------
Port topology(dax::cont::DataSet& Data )
{
  return Port(&Data, field_topology());
}

Port topology(const Port &port )
{
  return Port(port.dataSet(), field_topology());
}

template <typename T>
Port topology(const Filter<T>& filter)
{
  return topology(filter.outputPort());
}

#endif // PORTGENERATORS_H
