##=============================================================================
##
##  Copyright (c) Kitware, Inc.
##  All rights reserved.
##  See LICENSE.txt for details.
##
##  This software is distributed WITHOUT ANY WARRANTY; without even
##  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
##  PURPOSE.  See the above copyright notice for more information.
##
##  Copyright 2012 Sandia Corporation.
##  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
##  the U.S. Government retains certain rights in this software.
##
##=============================================================================

include_directories(${Boost_INCLUDE_DIRS})

set(headers
  ArrayContainerControl.h
  ArrayContainerControlBasic.h
  ArrayContainerControlImplicit.h
  ArrayHandle.h
  ArrayHandleConstant.h
  ArrayHandleCounting.h
  ArrayPortal.h
  Assert.h
  DeviceAdapter.h
  DeviceAdapterSerial.h
  Error.h
  ErrorControl.h
  ErrorControlAssert.h
  ErrorControlBadValue.h
  ErrorControlInternal.h
  ErrorControlOutOfMemory.h
  ErrorExecution.h
  GenerateInterpolatedCells.h
  GenerateTopology.h
  PermutationContainer.h
  Scheduler.h
  Timer.h
  UniformGrid.h
  UnstructuredGrid.h
  ${Dax_BINARY_DIR}/dax/cont/VectorOperations.h
  )
#-----------------------------------------------------------------------------
add_subdirectory(internal)

dax_declare_headers(${impl_headers} ${headers})


#-----------------------------------------------------------------------------
set(DAX_VECTOR_OPERATIONS_NAMESPACE "dax::cont")
set(DAX_VECTOR_OPERATIONS_NAMESPACE_NAME "dax_cont")
set(DAX_VECTOR_OPERATIONS_NAMESPACE_START "namespace dax { namespace cont {")
set(DAX_VECTOR_OPERATIONS_NAMESPACE_END   "} }")
set(DAX_VECTOR_OPERATIONS_EXPORT "DAX_CONT_EXPORT")
configure_file(
  ${Dax_SOURCE_DIR}/dax/internal/VectorOperations.h.in
  ${Dax_BINARY_DIR}/dax/cont/VectorOperations.h
  @ONLY
  )

add_subdirectory(arg)
add_subdirectory(scheduling)
add_subdirectory(sig)
add_subdirectory(testing)
