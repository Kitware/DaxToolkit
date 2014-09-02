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

set(worklets
  CellAverage.h
  CellDataToPointData.h
  CellGradient.h
  Cosine.h
  Elevation.h
  Magnitude.h
  MarchingCubes.h
  MarchingTetrahedra.h
  PointDataToCellData.h
  Sine.h
  Slice.h
  Square.h
  Tetrahedralize.h
  Threshold.h
  )

dax_declare_worklets(${worklets})

add_subdirectory(internal)
add_subdirectory(testing)
