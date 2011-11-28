#ifndef DATASET_H
#define DATASET_H

#include <vector>

#include "daxArray.h"

class DataSet
{
public:
  typedef dax::internal::BaseArray Field;
  typedef dax::Coordinates Coordinates;

  DataSet( const std::size_t& numPoints, const std::size_t& numCells);
  DataSet();
  virtual ~DataSet(){}

  std::size_t numPoints() const { return NumPoints; }
  std::size_t numCells() const { return NumCells; }

  std::size_t numPointFields() const { return PointFields.size(); }
  std::size_t numCellFields() const { return CellFields.size(); }

  bool addPointField(Field* field);
  bool addCellField(Field* field);

  Field* pointField(const std::string& n);
  Field* cellField(const std::string& n);

  virtual const Coordinates* points() const=0;

  bool removePointField(const std::string& n);
  bool removeCellField(const std::string& n);

protected:
  std::vector<Field*> PointFields;
  std::vector<Field*> CellFields;

  std::size_t NumPoints;
  std::size_t NumCells;

private:  
  bool addField(Field* f, std::vector<Field*>& fields);
  bool removeField(const std::string& n,std::vector<Field*>& fields);
  Field* getField(const std::string& n, std::vector<Field*>& fields);

  DataSet(const DataSet&);
  void operator=(const DataSet&);
};

//------------------------------------------------------------------------------
DataSet::DataSet( const std::size_t& numPoints, const std::size_t& numCells):
  NumPoints(numPoints), NumCells(numCells)
{
}

//------------------------------------------------------------------------------
DataSet::DataSet():
  NumPoints(0), NumCells(0)
{
}

//------------------------------------------------------------------------------
bool DataSet::addPointField(Field* field)
{
  return this->addField(field,this->PointFields);
}

//------------------------------------------------------------------------------
bool DataSet::addCellField(Field* field)
{
  return this->addField(field,this->CellFields);
}

//------------------------------------------------------------------------------
bool DataSet::addField(Field* field,std::vector<Field*>& fields)
{
  //need proper ownership of the array
  //currently we are copying it rather than properly using a smart pointer
  fields.push_back(field);
  return true;
}

//------------------------------------------------------------------------------
DataSet::Field* DataSet::pointField(const std::string& n)
{
  return this->getField(n,this->PointFields);
}

//------------------------------------------------------------------------------
DataSet::Field* DataSet::cellField(const std::string& n)
{
  return this->getField(n,this->CellFields);
}

//------------------------------------------------------------------------------
DataSet::Field* DataSet::getField(const std::string& n,
                                      std::vector<Field*>& fields)
{
  std::vector<Field*>::iterator i;
  for(i=fields.begin();i!=fields.end();++i)
    {
    if(n == (*i)->name())
      {
      return (*i);
      }
    }
  return NULL;
}

//------------------------------------------------------------------------------
bool DataSet::removePointField(const std::string& n)
{
  return this->removeField(n,this->PointFields);
}

//------------------------------------------------------------------------------
bool DataSet::removeCellField(const std::string& n)
{
  return this->removeField(n,this->CellFields);
}

//------------------------------------------------------------------------------
bool DataSet::removeField(const std::string& n, std::vector<Field*>& fields)
{
  std::vector<Field*>::iterator it;
  for(it=fields.begin();it!=fields.end();++it)
    {
    if ((*it)->name() == n)
      {
      fields.erase(it);
      return true;
      }
    }
  return false;
}



#endif // DATASET_H
