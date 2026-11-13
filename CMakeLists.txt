cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SlicerCBCTPlanner)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/HamiDeshoon/SlicerCBCTPlanner")
set(EXTENSION_CATEGORY "Dental")
set(EXTENSION_CONTRIBUTORS "Seyyed Hamid Aghlmand Sarmi")
set(EXTENSION_DESCRIPTION "3D Slicer extension for dental CBCT implant & surgical planning featuring DentalSegmentator 3D/Volume Rendering visualization, interactive dental arch curve generation, panoramic curved planar reformation (CPR), and synchronized cross-sectional slice visualization with axial intersection indicators.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/HamiDeshoon/SlicerCBCTPlanner/main/CBCTPlanner/Resources/Icons/CBCTPlanner.png")
set(EXTENSION_STATUS "Development")
set(EXTENSION_DEPENDS "NA")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(CBCTPlanner)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
