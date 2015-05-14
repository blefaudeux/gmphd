cmake_minimum_required (VERSION 2.6) 
project (GMPHD) 

# Recurse into the "GMPHD" and "demo" subdirectories. This does not actually 
# cause another cmake executable to run. The same process will walk through 
# the project's entire directory structure.
add_subdirectory (libGMPHD) 
add_subdirectory (demo) 
