# Define the project's name
project(LIBUKF)

cmake_minimum_required(VERSION 2.6)

# Add all the files we're interested in (mostly for QTCreator)
file(GLOB SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB HEADERS ${PROJECT_SOURCE_DIR}/headers/*.h)

# Define the included directories
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/headers)

# Try to find the needed packages
find_package(Boost REQUIRED)
find_package(eigen3)
if(NOT eigen3_FOUND)
  set(EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}")

  if(NOT EIGEN3_INCLUDE_DIR )
    # Try to manually define Eigen location
    set(ENV{EIGEN3_INCLUDE_DIR} "/usr/include/eigen3/")
    set(EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}")

    # Raise error if that's not enough
    if(NOT EIGEN3_INCLUDE_DIR )
      message(FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
    endif()

    include_directories( "${EIGEN3_INCLUDE_DIR}" )
  endif()
endif()

# GCC flags
SET( GCC_COVERAGE_COMPILE_FLAGS "-std=c++11 -Wall -Wextra -pedantic")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )

# Set debug or release configurations
if( CMAKE_BUILD_TYPE MATCHES "[Dd][Ee][Bb][Uu][Gg]" )
    message( "INFO: DEBUG BUILD" )
    # for debug type builds, turn on verbose makefiles
    SET(CMAKE_VERBOSE_MAKEFILE ON)

    # Tell other CMake files that we're doing a debug build
    SET( DEBUG_BUILD 1 )

    # Tell C/C++ that we're doing a debug build
    ADD_DEFINITIONS( -DDEBUG )
endif()

if( CMAKE_BUILD_TYPE MATCHES "[Rr][Ee][Ll][Ee][Aa][Ss][Ee]" )
    message( "INFO: RELEASE BUILD" )
endif()

# Create static and dynamic libraries
add_library (GMPHD_static STATIC ${SRC} ${HEADERS})
add_library (GMPHD_dynamic SHARED ${SRC} ${HEADERS})

# Set the link libraries :
TARGET_LINK_LIBRARIES(GMPHD_static ${Boost_LIBRARIES} )
TARGET_LINK_LIBRARIES(GMPHD_dynamic ${Boost_LIBRARIES} )

### Define the installation procedure
install(TARGETS GMPHD_dynamic GMPHD_static DESTINATION ${PROJECT_SOURCE_DIR}/lib)
install(FILES ${HEADERS} DESTINATION ${PROJECT_SOURCE_DIR}/include)
