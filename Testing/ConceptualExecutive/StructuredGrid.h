#ifndef STRUCTUREDGRID_H
#define STRUCTUREDGRID_H

#include "DataSet.h"

class StructuredGrid : public DataSet
{
public:  
  StructuredGrid(dax::Vector3 origin, dax::Vector3 spacing,
                 dax::Extent3 extents);

  virtual ~StructuredGrid();

  virtual const dax::Coordinates* points() const;

protected:
  dax::Vector3 computePointCoordinate(int index) const;
  dax::Id3 extentDimensions() const;
  dax::Id3 flatIndexToIndex3(dax::Id index) const;

  dax::Vector3 Origin;
  dax::Vector3 Spacing;
  dax::Extent3 Extent;

  dax::Coordinates* Coords;
};

//------------------------------------------------------------------------------
StructuredGrid::StructuredGrid(dax::Vector3 origin, dax::Vector3 spacing,
                               dax::Extent3 extents):
  DataSet(), Origin(origin), Spacing(spacing), Extent(extents)
{
  dax::Id3 size = this->extentDimensions();
  this->NumPoints = size.x * size.y * size.z;

  size = size - dax::make_Id3(1,1,1);
  this->NumCells = size.x * size.y * size.z;

  this->Coords = new dax::Coordinates();
  this->Coords->reserve(this->NumPoints);
  for(std::size_t i=0; i < this->NumPoints; ++i)
    {
    this->Coords->push_back(this->computePointCoordinate(i));
    }
}

//------------------------------------------------------------------------------
StructuredGrid::~StructuredGrid()
{
  delete this->Coords;
}

//------------------------------------------------------------------------------
const dax::Coordinates* StructuredGrid::points() const
{
  //returns the dataArray that is the point coordinates
  return this->Coords;
}

//------------------------------------------------------------------------------
dax::Vector3 StructuredGrid::computePointCoordinate(int index) const
{
  dax::Id3 ijk = this->flatIndexToIndex3(index);
  return this->Origin + ijk * this->Spacing;
}

//------------------------------------------------------------------------------
dax::Id3 StructuredGrid::extentDimensions() const
{
  return this->Extent.Max - this->Extent.Min + dax::make_Id3(1, 1, 1);;
}

//------------------------------------------------------------------------------
dax::Id3 StructuredGrid::flatIndexToIndex3(dax::Id index) const
  {
    dax::Id3 dims = this->extentDimensions();
    dax::Id3 ijk;
    ijk.x = index % dims.x;
    ijk.y = (index / dims.x) % dims.y;
    ijk.z = (index / (dims.x * dims.y));

    return ijk + this->Extent.Min;
  }

#endif // STRUCTUREDGRID_H
