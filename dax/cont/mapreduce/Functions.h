#ifndef __dax_exec_mapreduce_Functions_h
#define __dax_exec_mapreduce_Functions_h

#include <dax/cont/UniformGrid.h>
#include <dax/exec/Cell.h>

#include <iostream>

namespace dax {
namespace cont {
namespace mapreduce {

/// Returns the number of cells in the grid type passed in
/// Expects that template parameter T is a control data structure
template <typename T>
dax::Id num_cells(const T& t)
  {
  return t.GetNumberOfCells();
  }

/// Returns the number of points in the grid type passed in
/// Expects that template parameter T is a control data structure
template <typename T>
dax::Id num_points(const T& t)
  {
  return t.GetNumberOfPoints();
  }

/// Returns the total topology size, which is number of points per cell
/// times the number of cells.
/// Expects that template parameter T is a control data structure.
template <typename T>
dax::Id topology_size(const T& t);

/// Returns the total topology size, which is number of points per cell
/// times the number of cells.
/// Specialization for Uniform Grid, as currently we can't get cell type
/// from the grid. ToDo: Add a trait to the grid that is the cell type
template <>
dax::Id topology_size(const dax::cont::UniformGrid& t)
  {
  return 8 * t.GetNumberOfCells();
  }

/// Sorts t in place in ascending order. The sort is not guaranteed to keep
/// duplicate enteries in the same order as they appear.
/// { 5, 8 , 2, 0 } will be sorted to be { 0, 2, 5, 8 }
template <typename T>
void sort(T& t);

/// Counts the number occurances of each value in T. the result of
/// of the counting is placed in U
template <typename T, typename U>
void count_occurances(T &t, U &u);

} //mapreduce
} //cont
} //dax


#endif // __dax_cont_mapreduce_Functions_h
