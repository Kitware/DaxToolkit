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

set(headers
  ArrayPortalFromIterators.h
  DerivativeWeights.h
  ErrorMessageBuffer.h
  FieldAccess.h
  Functor.h
  GridTopologies.h
  InterpolationWeights.h
  ${Dax_BINARY_DIR}/dax/exec/internal/Members.h
  TopologyUniform.h
  TopologyUnstructured.h
  WorkletBase.h
  )

dax_declare_headers(${headers})

set(DAX_MEMBERS_CONFIGURED_INCLUDE_PATH "dax/exec/internal")
set(DAX_MEMBERS_NAMESPACE "dax::exec::internal")
set(DAX_MEMBERS_NAMESPACE_NAME "dax_exec_internal")
set(DAX_MEMBERS_NAMESPACE_START "namespace dax { namespace exec { namespace internal {")
set(DAX_MEMBERS_NAMESPACE_END   "} } }")
set(DAX_MEMBERS_EXPORT "DAX_EXEC_EXPORT")
configure_file(
  ${Dax_SOURCE_DIR}/dax/internal/Members.h.in
  ${Dax_BINARY_DIR}/dax/exec/internal/Members.h
  @ONLY
  )

add_subdirectory(kernel)
add_subdirectory(testing)

