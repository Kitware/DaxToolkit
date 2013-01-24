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
  TopologyUniform.h
  TopologyUnstructured.h
  WorkletBase.h
  FlatIndex.h
  IJKIndex.h
  )

dax_declare_headers(${headers})

add_subdirectory(kernel)
add_subdirectory(testing)

