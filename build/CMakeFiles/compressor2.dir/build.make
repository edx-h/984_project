# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /root/gpu_db/984_project

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /root/gpu_db/984_project/build

# Include any dependencies generated for this target.
include CMakeFiles/compressor2.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/compressor2.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/compressor2.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/compressor2.dir/flags.make

CMakeFiles/compressor2.dir/comp/compressor2.cu.o: CMakeFiles/compressor2.dir/flags.make
CMakeFiles/compressor2.dir/comp/compressor2.cu.o: ../comp/compressor2.cu
CMakeFiles/compressor2.dir/comp/compressor2.cu.o: CMakeFiles/compressor2.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/root/gpu_db/984_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/compressor2.dir/comp/compressor2.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/compressor2.dir/comp/compressor2.cu.o -MF CMakeFiles/compressor2.dir/comp/compressor2.cu.o.d -x cu -c /root/gpu_db/984_project/comp/compressor2.cu -o CMakeFiles/compressor2.dir/comp/compressor2.cu.o

CMakeFiles/compressor2.dir/comp/compressor2.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/compressor2.dir/comp/compressor2.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/compressor2.dir/comp/compressor2.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/compressor2.dir/comp/compressor2.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target compressor2
compressor2_OBJECTS = \
"CMakeFiles/compressor2.dir/comp/compressor2.cu.o"

# External object files for target compressor2
compressor2_EXTERNAL_OBJECTS =

compressor2: CMakeFiles/compressor2.dir/comp/compressor2.cu.o
compressor2: CMakeFiles/compressor2.dir/build.make
compressor2: CMakeFiles/compressor2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/root/gpu_db/984_project/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable compressor2"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/compressor2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/compressor2.dir/build: compressor2
.PHONY : CMakeFiles/compressor2.dir/build

CMakeFiles/compressor2.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/compressor2.dir/cmake_clean.cmake
.PHONY : CMakeFiles/compressor2.dir/clean

CMakeFiles/compressor2.dir/depend:
	cd /root/gpu_db/984_project/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /root/gpu_db/984_project /root/gpu_db/984_project /root/gpu_db/984_project/build /root/gpu_db/984_project/build /root/gpu_db/984_project/build/CMakeFiles/compressor2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/compressor2.dir/depend

