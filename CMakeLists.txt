cmake_minimum_required (VERSION 2.6) 
project (GMPHD) 

SET(DEMO         FALSE     CACHE  BOOL    "Build the demo app")

add_subdirectory (libGMPHD) 

if(DEMO)
	add_subdirectory (demo) 
endif()
