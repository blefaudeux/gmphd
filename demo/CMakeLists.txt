# Define the project's name
project(Demo)

cmake_minimum_required(VERSION 2.6)

# Make sure the compiler can find include files from our GMPHD library. 
# include_directories(${PROJECT_SOURCE_DIR}/src)
# include_directories(${PROJECT_SOURCE_DIR}/headers)
# include_directories(${PROJECT_SOURCE_DIR}/../libGMPHD/headers)
# include_directories(${PROJECT_SOURCE_DIR}/../libGMPHD/lib)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

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
  endif()
  include_directories(${EIGEN3_INCLUDE_DIR})
  MESSAGE("Eigen path : ${EIGEN3_INCLUDE_DIR}\n")
endif()

find_package( OpenCV REQUIRED )

# Set debug/release flags
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

# Add all the files we're interested in (mostly for QTCreator)
file(GLOB SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB HEADERS ${PROJECT_SOURCE_DIR}/headers/*.h)

MESSAGE("Cpp files : \n ${SRC}\n")
MESSAGE("H files : \n ${HEADERS}\n")

# Make sure the linker can find the GMPHD library once it is built. 
set(LIBGMPHD_BINARY_DIR ${PROJECT_SOURCE_DIR}/../libGMPHD/lib)

link_directories (${LIBGMPHD_BINARY_DIR}) 
MESSAGE("LibGMPHD Path :\n ${LIBGMPHD_BINARY_DIR}\n")

# Add executable "demo"
add_executable(demo ${SRC} ${HEADERS})

# Link the executable to the GMPHD library.
target_link_libraries (demo GMPHD_dynamic ${OpenCV_LIBS})
