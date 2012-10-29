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

#-----------------------------------------------------------------------------
if (DAX_ENABLE_OPENMP)
  dax_configure_device(OpenMP)
endif (DAX_ENABLE_OPENMP)

if (DAX_ENABLE_CUDA)
  dax_configure_device(Cuda)
endif (DAX_ENABLE_CUDA)

if (DAX_ENABLE_TBB)
  dax_configure_device(TBB)
endif (DAX_ENABLE_TBB)

#-----------------------------------------------------------------------------
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/FY11TimingConfig.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/FY11TimingConfig.h
  @ONLY
  )

#-----------------------------------------------------------------------------
set(headers
  Pipeline.h
  )

set(sources
  #PipelineDebug.cxx
  #FY11Timing.cxx
  ArgumentsParser.cxx
  Timer.cxx
  )

#-----------------------------------------------------------------------------
add_executable(FY11TimingSerial ${sources} ${headers} PipelineSerial.cxx)
set_source_files_properties(${headers} PROPERTIES HEADER_FILE_ONLY TRUE)
target_link_libraries(FY11TimingSerial
  ${Boost_LIBRARIES}
  )

add_test(FY11TimingSerial1-128
  ${EXECUTABLE_OUTPUT_PATH}/FY11TimingSerial --pipeline 1 --size 128)
add_test(FY11TimingSerial2-128
  ${EXECUTABLE_OUTPUT_PATH}/FY11TimingSerial --pipeline 2 --size 128)
add_test(FY11TimingSerial3-128
  ${EXECUTABLE_OUTPUT_PATH}/FY11TimingSerial --pipeline 3 --size 128)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_OPENMP)
  add_executable(FY11TimingOpenMP ${sources} ${headers} PipelineOpenMP.cxx)
  target_link_libraries(FY11TimingOpenMP
    ${Boost_LIBRARIES}
    )

  add_test(FY11TimingOpenMP1-128
    ${EXECUTABLE_OUTPUT_PATH}/FY11TimingOpenMP --pipeline 1 --size 128)
  add_test(FY11TimingOpenMP2-128
    ${EXECUTABLE_OUTPUT_PATH}/FY11TimingOpenMP --pipeline 2 --size 128)
  add_test(FY11TimingOpenMP3-128
    ${EXECUTABLE_OUTPUT_PATH}/FY11TimingOpenMP --pipeline 3 --size 128)
endif (DAX_ENABLE_OPENMP)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_TBB)
  add_executable(FY11TimingTBB ${sources} ${headers} PipelineTBB.cxx)
  target_link_libraries(FY11TimingTBB
    ${Boost_LIBRARIES}
    )

  add_test(FY11TimingTBB1-128
    ${EXECUTABLE_OUTPUT_PATH}/FY11TimingTBB --pipeline 1 --size 128)
  add_test(FY11TimingTBB2-128
    ${EXECUTABLE_OUTPUT_PATH}/FY11TimingTBB --pipeline 2 --size 128)
  add_test(FY11TimingTBB3-128
    ${EXECUTABLE_OUTPUT_PATH}/FY11TimingTBB --pipeline 3 --size 128)
endif (DAX_ENABLE_TBB)

#-----------------------------------------------------------------------------
if (DAX_ENABLE_CUDA)
  dax_disable_troublesome_thrust_warnings()
  cuda_add_executable(FY11TimingCuda
    ${sources} ${headers} PipelineCuda.h PipelineCuda.cu
    )
  set_source_files_properties(PipelineCuda.h PROPERTIES HEADER_FILE_ONLY TRUE)
  target_link_libraries(FY11TimingCuda
    ${Boost_LIBRARIES}
    )

  add_test(FY11TimingCuda1-128
    ${EXECUTABLE_OUTPUT_PATH}/FY11TimingCuda --pipeline 1 --size 128)
  add_test(FY11TimingCuda2-128
    ${EXECUTABLE_OUTPUT_PATH}/FY11TimingCuda --pipeline 2 --size 128)
  add_test(FY11TimingCuda3-128
    ${EXECUTABLE_OUTPUT_PATH}/FY11TimingCuda --pipeline 3 --size 128)
endif (DAX_ENABLE_CUDA)


#-----------------------------------------------------------------------------
#add_test(FY11Timing1-256
#  ${EXECUTABLE_OUTPUT_PATH}/FY11Timing --pipeline 1 --size 256)
#add_test(FY11Timing2-256
#  ${EXECUTABLE_OUTPUT_PATH}/FY11Timing --pipeline 2 --size 256)
#add_test(FY11Timing3-256
#  ${EXECUTABLE_OUTPUT_PATH}/FY11Timing --pipeline 3 --size 256)
