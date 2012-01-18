#=========================================================================
#
#  This software is distributed WITHOUT ANY WARRANTY; without even
#  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#  PURPOSE.  See the above copyright notice for more information.
#
#=========================================================================

set(headers
  Cell.h
  Derivative.h
  Field.h
  Interpolate.h
  ParametricCoordinates.h
  WorkMapCell.h
  WorkMapField.h
  WorkMapReduceCell.h

  ${Dax_BINARY_DIR}/dax/exec/VectorOperations.h
)

dax_declare_headers(${headers})

#-----------------------------------------------------------------------------
set(DAX_VECTOR_OPERATIONS_NAMESPACE_NAME "dax_exec")
set(DAX_VECTOR_OPERATIONS_NAMESPACE_START "namespace dax { namespace exec {")
set(DAX_VECTOR_OPERATIONS_NAMESPACE_END   "} }")
set(DAX_VECTOR_OPERATIONS_EXPORT "DAX_EXEC_EXPORT")
configure_file(
  ${Dax_SOURCE_DIR}/dax/internal/VectorOperations.h.in
  ${Dax_BINARY_DIR}/dax/exec/VectorOperations.h
  @ONLY
  )

#-----------------------------------------------------------------------------
add_subdirectory(mapreduce)
add_subdirectory(internal)

set(unit_tests
  UnitTestCells.cxx
  UnitTestDerivative.cxx
  UnitTestInterpolate.cxx
  UnitTestParametricCoordinates.cxx
  UnitTestWorkMapCell.cxx
  UnitTestWorkMapField.cxx
  )
dax_unit_tests(SOURCES ${unit_tests})
