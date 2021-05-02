cmake_minimum_required (VERSION 2.6)

project(gmphd)

SET(DEMO         FALSE     CACHE  BOOL    "Build the demo app")

add_library (gmphd INTERFACE)
target_include_directories(gmphd INTERFACE "gmphd/.")

if(DEMO)
	add_subdirectory (demo)
endif()
