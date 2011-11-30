#ifndef __dax_cont_StructuredGrid_h
#define __dax_cont_StructuredGrid_h

#include <dax/cont/DataSet.h>

namespace dax { namespace cont {

class StructuredGrid :
    public dax::cont::DataSet
{
private:
  struct Extent3 {
    Id3 Min;
    Id3 Max;
  };

public:  
  StructuredGrid();
  StructuredGrid(const dax::Vector3 &origin,
                 const dax::Vector3 &spacing,
                 const dax::Id3 &minExtents,
                 const dax::Id3 &maxExtents);

  Vector3 Origin;
  Vector3 Spacing;
  Extent3 Extent;

  virtual const Coordinates* points() const { return NULL; }

  virtual ~StructuredGrid();

private:
  dax::Id3 extentDimensions()
  {
    return Extent.Max - Extent.Min + dax::make_Id3(1, 1, 1);
  }

  dax::Id numberOfPoints()
  {
    dax::Id3 dims = this->extentDimensions();
    return dims.x*dims.y*dims.z;
  }

  dax::Id numberOfCells()
  {
    dax::Id3 dims = this->extentDimensions() - dax::make_Id3(1, 1, 1);
    return dims.x*dims.y*dims.z;
  }


};

//------------------------------------------------------------------------------
StructuredGrid::StructuredGrid():
  DataSet()
{

}

//------------------------------------------------------------------------------
StructuredGrid::StructuredGrid(const dax::Vector3 &origin,
                               const dax::Vector3 &spacing,
                               const dax::Id3 &minExtents,
                               const dax::Id3 &maxExtents):
  DataSet()
{
  this->Origin = origin;
  this->Spacing = spacing;
  this->Extent.Min = minExtents;
  this->Extent.Max = maxExtents;

  this->NumPoints = this->numberOfPoints();
  this->NumCells = this->numberOfCells();
}

//------------------------------------------------------------------------------
StructuredGrid::~StructuredGrid()
{
}


} }
#endif // __dax_cont_StructuredGrid_h
