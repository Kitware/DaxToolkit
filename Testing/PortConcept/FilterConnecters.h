#ifndef FILTERCONNECTERS_H
#define FILTERCONNECTERS_H

#include "Concept.h"

namespace{
Port makePort(dax::cont::DataSet* ds,
              const std::string &name,
              const field_cells &fc)
{
  return Port(ds,ds->cellField(name),fc);
}

Port makePort(dax::cont::DataSet* ds,
              const std::string &name,
              const field_points &fc)
{
  return Port(ds,ds->pointField(name),fc);
}
}

template <typename Source>
class FilterConnector
{
public:
  FilterConnector(Source *const f, const Port& p):
  Port_(p),
  Source_(f)
  {}

  template<typename FieldType>
  FilterConnector(const FilterConnector &fc,
                  const std::string& name,
                  const FieldType& fieldType):
  Port_(makePort(fc.port().dataSet(),name,fieldType)),
  Source_(fc.source())
  {
  }

  template<typename FieldType>
  FilterConnector(const FilterConnector &fc,
                  const FieldType& fieldType):
  Port_(fc.port().dataSet(),fieldType),
  Source_(fc.source())
  {}


  Port port() const
  {
  return Port_;
  }

  Source * const source() const
  {
  return Source_;
  }

private:
  Port Port_;
  Source* const Source_; //constant pointer to changle Source
};

#endif // FILTERCONNECTERS_H
