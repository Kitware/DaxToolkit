#ifndef __dax_cont_internal_ExtractTopology_h
#define __dax_cont_internal_ExtractTopology_h

#include <dax/Types.h>
#include <dax/exec/Cell.h>
#include <dax/exec/Field.h>
#include <dax/exec/WorkMapCell.h>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

namespace dax {
namespace exec {
namespace kernel {
namespace internal {


template <typename CellType>
DAX_WORKLET void ExtractTopology(dax::exec::WorkMapCell<CellType> work,
                                 dax::Id newIndex,
                                 dax::exec::Field<dax::Id> &topology)
  {
  typedef typename CellType::PointIds PointIds;
  CellType cell(work.GetCell());
  dax::Id offset(CellType::NUM_POINTS * newIndex);

  //manual unrolled in an attempt to make this faster
  PointIds pointIndices = cell.GetPointIndices();

  dax::Id* temp = topology.GetArray().GetPointer()+offset;
  for(dax::Id i=0;i < CellType::NUM_POINTS; ++i)
    {
    temp[i] = pointIndices[i];
    }
  }

template<class CellType>
struct ExtractTopologyParameters
{
  typename CellType::TopologyType grid;
  dax::exec::Field<dax::Id> outField;
  dax::exec::Field<dax::Id> subset;
};

template<class CellType>
struct ExtractTopologyFunctor {
  DAX_EXEC_EXPORT void operator()(
      ExtractTopologyParameters<CellType> &parameters,
      dax::Id index, const dax::exec::internal::ErrorHandler &errorHandler)
  {
    dax::exec::WorkMapCell<CellType> work(parameters.grid, errorHandler);
    work.SetCellIndex(dax::exec::internal::fieldAccessNormalGet(
                        parameters.subset,index));
    dax::exec::kernel::internal::ExtractTopology(work,index,
                                                 parameters.outField);
  }
};

}
}
}
} //dax::exec::kernel::internal

namespace dax {
namespace cont {
namespace internal {

template<typename DeviceAdapter, typename GridType>
class ExtractTopology
{
public:
  /// Extract a subset of the cells topology. cellsToExtract contain
  /// The cell ids to extract. The resulting array contains only
  /// the point ids for each cell, so to the point ids of the third
  /// cell would be in positions CellType::NUM_POINTS * 3 to (CellType::NUM_POINTS * 4) - 1.
  /// By default the extracted topology use the original point ids. If
  /// /p OrderedUniqueValueIndices is set to true the id's are modified to go from 0, N where
  /// N is the number of point Ids. This operation is stable in that the original relative
  /// ordering of the point Ids is kept intact.
  ExtractTopology(const GridType& grid,
                  dax::cont::ArrayHandle<dax::Id,DeviceAdapter> &cellsToExtract,
                  bool OrderedUniqueValueIndices=false)
    {
    typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
    typedef typename GridPackageType::ExecutionCellType CellType;

    //verify the input
    DAX_ASSERT_CONT(grid.GetNumberOfCells() > 0);
    DAX_ASSERT_CONT(cellsToExtract.GetNumberOfEntries() > 0);

    this->DoExtract(grid,cellsToExtract);

    if(OrderedUniqueValueIndices &&
       this->Topology.GetNumberOfEntries() > CellType::NUM_POINTS)
      {
      dax::cont::ArrayHandle<dax::Id,DeviceAdapter> temp(
            this->Topology.GetNumberOfEntries());
      DeviceAdapter::Copy(this->Topology,temp);
      DeviceAdapter::Sort(temp);
      DeviceAdapter::Unique(temp);
      DeviceAdapter::LowerBounds(temp,this->Topology,this->Topology);
      }

    this->Topology.CompleteAsOutput();
    }

  /// Returns an array handle to the execution enviornment data that
  /// contians the extracted topology
  dax::cont::ArrayHandle<dax::Id,DeviceAdapter>& GetTopology(){return Topology;}

private:
  void DoExtract(const GridType& grid,
                 dax::cont::ArrayHandle<dax::Id,DeviceAdapter> &cellsToExtract);

  dax::cont::ArrayHandle<dax::Id,DeviceAdapter> Topology;
};

//-----------------------------------------------------------------------------
template<typename DeviceAdapter, typename GridType>
inline void ExtractTopology<DeviceAdapter,GridType>::DoExtract(
    const GridType& grid,
    dax::cont::ArrayHandle<dax::Id,DeviceAdapter> &cellsToExtract)
{
  typedef dax::cont::internal::ExecutionPackageGrid<GridType> GridPackageType;
  typedef typename GridPackageType::ExecutionCellType CellType;

  //construct the input grid
  GridPackageType inPGrid(grid);

  //construct the topology result array
  const dax::Id numCellsToExtract(cellsToExtract.GetNumberOfEntries());
  const dax::Id size(numCellsToExtract * CellType::NUM_POINTS);

  this->Topology = dax::cont::ArrayHandle<dax::Id,DeviceAdapter>(size);

  //we want the size of the points to be based on the numCells * points per cell
  dax::cont::internal::ExecutionPackageFieldOutput<dax::Id,DeviceAdapter>
      result(this->Topology, size);

  //package up the cells to extract so we can lookup the right ids
  dax::cont::internal::ExecutionPackageFieldInput<dax::Id,DeviceAdapter>
      cellsExtractPackage(cellsToExtract, numCellsToExtract);

  //construct the parameters list for the function
  dax::exec::kernel::internal::ExtractTopologyParameters<CellType> etParams =
                                      {
                                      inPGrid.GetExecutionObject(),
                                      result.GetExecutionObject(),
                                      cellsExtractPackage.GetExecutionObject()
                                      };
  DeviceAdapter::Schedule(
    dax::exec::kernel::internal::ExtractTopologyFunctor<CellType>(),
    etParams,numCellsToExtract);
}


}
}
}
#endif // __dax_exec_internal_ExtractTopology_h
