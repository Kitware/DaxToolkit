#ifndef __dax_exec_mapreduce_RemoveCell_h
#define __dax_exec_mapreduce_RemoveCell_h

#include <dax/Types.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>

#include <dax/internal/GridStructures.h>
#include <dax/exec/internal/FieldAccess.h>

#include <dax/cont/mapreduce/Functions.h>
#include <dax/cont/mapreduce/Classify.h>

namespace dax {
namespace cont {
namespace mapreduce {

class RemoveCell
{

  typedef dax::cont::mapreduce::num_cells ClassifySize;
  typedef dax::cont::mapreduce::stencil Scan;
  typedef dax::cont::mapreduce::no_size GenerateSize;
  typedef dax::cont::mapreduce::extract Generate;


  template<typename InGridType, typename HandleType, typename OutGridType>
  void run(const InGridType& inGrid,
           HandleType& handle,
           OutGridType& outGrid)
    {
    //call size
    //call worklet
    //call scan
    //call generateSize
    //call Generate
    }

  template<typename InGridType, typename OutGridType>
  void run(const InGridType& inGrid,
           OutGridType& outGrid)
    {
    //call size
    //call worklet
    //call scan
    //call generateSize
    //call Generate
    }

  template<typename GridType, typename HandleType>
  void callWorklet(const GridType& geom, HandleType &handle)
  {
    return static_cast<Derived*>(this)->Worklet(geom,handle);
  }
  template<typename GridType>
  void callWorklet(const GridType& geom)
  {
    return static_cast<Derived*>(this)->Worklet(geom);
  }

private:
  dax::Id WorkletSize;

};



} //mapreduce
} //exec
} //dax


#endif // __dax_exec_mapreduce_RemoveCell_h
