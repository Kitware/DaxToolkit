cmake_minimum_required(VERSION 2.8)
project (Functors)

# List of Boost features used:
# * Smart Ptr
find_package(Boost REQUIRED)

add_library(fncFramework
  fncObject.h
  fncObject.cxx
)

add_executable(fncTest
  fncTest.cxx)

target_link_libraries(fncTest
  fncFramework)
