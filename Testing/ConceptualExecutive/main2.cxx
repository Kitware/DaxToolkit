
#include "Filter.h"
#include "Worklets.h"

#include "StructuredGrid.h"

void createGrid(StructuredGrid *&grid)
{
  dax::Vector3 origin = dax::make_Vector3(0,0,0);
  dax::Vector3 spacing = dax::make_Vector3(.2,.2,.2);
  dax::Extent3 extents = dax::Extent3( dax::make_Id3(0,0,0), dax::make_Id3(10,10,10));
  grid = new StructuredGrid(origin,spacing,extents);
}

void buildExamplePipeline()
{

  std::cout << "Build Example Pipeline" << std::endl;
  StructuredGrid* grid;
  createGrid(grid);

  Filter<worklets::Elevation> elev(grid->points());
  Filter<worklets::Sine> sin(elev);
  Filter<worklets::Square> sq(sin);
  Filter<worklets::Cosine> cos(sq);

  cos.run();

  Filter<worklets::Square> sqa2(sq);

  sqa2.run();


}

int main()
{
  buildExamplePipeline();

  return 0;
}
