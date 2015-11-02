# Define the project's name
project(Demo)

cmake_minimum_required(VERSION 2.6)

# Make sure the compiler can find include files from our GMPHD library.
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/headers)
include_directories(${PROJECT_SOURCE_DIR}/../libGMPHD/headers)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Try to find the needed packages
find_package( OpenCV REQUIRED )
find_package( PkgConfig )
pkg_check_modules( EIGEN3 REQUIRED eigen3 )
include_directories( ${EIGEN3_INCLUDE_DIRS} )

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

# we use C++11 features
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()

if( CMAKE_BUILD_TYPE MATCHES "[Rr][Ee][Ll][Ee][Aa][Ss][Ee]" )
    message( "INFO: RELEASE BUILD" )
endif()

# Add all the files we're interested in (mostly for QTCreator)
file(GLOB SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB HEADERS ${PROJECT_SOURCE_DIR}/headers/*.h)

# Make sure the linker can find the GMPHD library once it is built.
set(LIBGMPHD_BINARY_DIR ${PROJECT_BINARY_DIR}/../libGMPHD)
link_directories (${LIBGMPHD_BINARY_DIR})

# Build the lib before the demo
add_executable(demo ${SRC} ${HEADERS})
add_dependencies(demo GMPHDs)

# Link the executable to the GMPHD library.
target_link_libraries (demo ${LIBGMPHD_BINARY_DIR}/libGMPHDs.a ${OpenCV_LIBS})
