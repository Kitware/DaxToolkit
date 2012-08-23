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
  Extent.h
  Functional.h
  Types.h
  TypeTraits.h
  VectorTraits.h


  ${Dax_BINARY_DIR}/dax/VectorOperations.h
  )

dax_declare_headers(${headers})

#-----------------------------------------------------------------------------
set(DAX_VECTOR_OPERATIONS_NAMESPACE "dax")
set(DAX_VECTOR_OPERATIONS_NAMESPACE_NAME "dax")
set(DAX_VECTOR_OPERATIONS_NAMESPACE_START "namespace dax {")
set(DAX_VECTOR_OPERATIONS_NAMESPACE_END   "}")
set(DAX_VECTOR_OPERATIONS_EXPORT "DAX_EXEC_CONT_EXPORT")
configure_file(
  ${Dax_SOURCE_DIR}/dax/internal/VectorOperations.h.in
  ${Dax_BINARY_DIR}/dax/VectorOperations.h
  @ONLY
  )

#-----------------------------------------------------------------------------
add_subdirectory(internal)

add_subdirectory(math)

add_subdirectory(exec)
add_subdirectory(cont)

if (DAX_ENABLE_THRUST)
  add_subdirectory(thrust)
endif (DAX_ENABLE_THRUST)

if (DAX_ENABLE_CUDA)
  add_subdirectory(cuda)
endif (DAX_ENABLE_CUDA)

if (DAX_ENABLE_OPENMP)
  add_subdirectory(openmp)
endif (DAX_ENABLE_OPENMP)

add_subdirectory(Testing)
