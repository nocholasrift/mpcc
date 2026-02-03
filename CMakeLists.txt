cmake_minimum_required(VERSION 3.5.0)
project(mpcc C CXX)

# add cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(AcadosUtils)


## Try to find catkin (optional)
find_package(catkin QUIET COMPONENTS
  geometry_msgs
  roscpp
  rospy
  std_msgs
  trajectory_msgs
  grid_map_ros
  costmap_2d
  amrl_logging
)

find_package(pybind11 QUIET)
find_package(Python3 REQUIRED)
find_package(ACADOS REQUIRED)
find_package(BLASFEO REQUIRED)
find_package(HPIPM REQUIRED)
find_package(Eigen3 REQUIRED)


add_subdirectory(third_party/HiGHS)

message(STATUS "ACADOS_INCLUDE_DIRS: ${ACADOS_INCLUDE_DIRS}")

########################
# Catkin-specific stuff
########################
if(catkin_FOUND)
  add_definitions(-DFOUND_CATKIN)
  message(STATUS "Catkin found, building ROS nodes")

  add_message_files(
    DIRECTORY
    FILES
      RLState.msg
  )

  add_service_files(
    FILES
      QuerySAC.srv
      QuerySACDI.srv
  )

  generate_messages(
    DEPENDENCIES 
      nav_msgs
      geometry_msgs 
      trajectory_msgs
  )

  catkin_package(
    CATKIN_DEPENDS message_runtime
  )
else()
  message(STATUS "Catkin not found, skipping ROS message/service generation")
endif()

set(HiGHS_INCLUDE_DIRS
  ${HIGHS_BINARY_DIR}
  third_party/autodiff
  third_party/HiGHS/app
  third_party/HiGHS/extern
  third_party/HiGHS/extern/zstr
  third_party/HiGHS/highs
  third_party/HiGHS/highs/io
  third_party/HiGHS/highs/pdlp/cupdlp
  third_party/HiGHS/highs/ipm/ipx
  third_party/HiGHS/highs/ipm/basiclu
  third_party/HiGHS/highs/lp_data
  third_party/HiGHS/highs/mip
  third_party/HiGHS/highs/model
  third_party/HiGHS/highs/presolve
  third_party/HiGHS/highs/qpsolver
  third_party/HiGHS/highs/simplex
  third_party/HiGHS/highs/test_kkt
  third_party/HiGHS/highs/util
)

if (APPLE)
  set(EIGEN3_INCLUDE_DIRS /opt/homebrew/opt/eigen/include/eigen3)
endif()

########################
# Common include paths
########################
include_directories(
  include
  ${HPIPM_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${ACADOS_INCLUDE_DIRS}
  ${BLASFEO_INCLUDE_DIRS}
  ${cpg_include}
)

if(catkin_FOUND)
  include_directories(${catkin_INCLUDE_DIRS})
endif()

########################
# ACADOS model generation
########################
set(ACADOS_MODELS
  unicycle_model_mpcc
  double_integrator_mpcc
)

set(ACADOS_OUTPUT_DIRS "")
set(ACADOS_OUTPUT_SRC_CXX "")
set(ACADOS_OUTPUT_HEADERS "")
set(ACADOS_OUTPUT_LIBS "")
set(ACADOS_TARGETS "")

set(ACADOS_CODE_DIR
    ${CMAKE_BINARY_DIR}/acados_code_gen
)
set(CASADI_GEN_DIR
  ${ACADOS_CODE_DIR}/casadi
)

foreach(model ${ACADOS_MODELS})
  set(${model}_SCRIPTS 
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/${model}/mpcc_model.py
    ${CMAKE_CURRENT_SOURCE_DIR}/scripts/${model}/mpcc_ocp.py
  )
  set(ACADOS_GEN_DIR
    ${ACADOS_CODE_DIR}/${model}
  )
  list(APPEND ACADOS_OUTPUT_DIRS
    # ${CMAKE_CURRENT_SOURCE_DIR}/scripts/${model}/c_generated_code 
    # ${CMAKE_CURRENT_SOURCE_DIR}/scripts/${model}/cpp_generated_code
    ${ACADOS_GEN_DIR}
    ${CASADI_GEN_DIR}
  )

  collect_acados_generated_code(${model} ${ACADOS_GEN_DIR} ${CASADI_GEN_DIR})
  list(APPEND ACADOS_OUTPUT_SRC_CXX ${${model}_SRC_CXX})
  list(APPEND ACADOS_OUTPUT_SRC ${${model}_SRC})
  list(APPEND ACADOS_OUTPUT_HEADERS ${${model}_HEADERS})
  list(APPEND ACADOS_OUTPUT_LIBS ${${model}_LIBS})

  message(${ACADOS_OUTPUT_LIBS})

  add_custom_command(
    OUTPUT ${${model}_OUTPUT_FILES}
    COMMAND echo "y" | ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/${model}/mpcc_ocp.py --yaml ${CMAKE_CURRENT_SOURCE_DIR}/params/${model}.yaml --output_dir ${ACADOS_GEN_DIR} --casadi_dir ${CASADI_GEN_DIR}
    DEPENDS ${ACADOS_SCRIPTS} ${CMAKE_CURRENT_SOURCE_DIR}/params/${model}.yaml
    COMMENT "Building MPC code with Acados"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/scripts/${model}
    VERBATIM
  )

  add_custom_target(
    GenerateAcadosCode_${model}
    DEPENDS ${${model}_OUTPUT_FILES} ${${model}_SCRIPTS} ${CMAKE_CURRENT_SOURCE_DIR}/params/${model}.yaml
  )
  list(APPEND ACADOS_TARGETS GenerateAcadosCode_${model})
endforeach()

message(WARNING "include dir: ${EIGEN3_INCLUDE_DIRS}")

########################
# Core MPCC library
########################
# add_library(cvxpy_lib STATIC ${cpg_src})
# target_compile_options(cvxpy_lib PRIVATE -O3 -w)
# target_include_directories(cvxpy_lib PRIVATE ${cpg_include})

########################
# Python bindings
########################
if (pybind11_FOUND)
  message(STATUS "Pybind11 found, building mpcc python bindings")
  add_definitions(-DFOUND_PYBIND11)

  find_package(PythonLibs REQUIRED)
  include_directories(${PYTHON_INCLUDE_DIRS})

  pybind11_add_module(py_mpcc
    src/mpcc/mpcc_core.cpp
    src/mpcc/mpcc_acados.cpp
    src/mpcc/mpcc_di_acados.cpp
    src/mpcc/py_mpcc.cpp
    # ${double_integrator_SRC_CXX}
  )
  add_dependencies(py_mpcc ${ACADOS_TARGETS})

  target_include_directories(py_mpcc PRIVATE 
    ${ACADOS_OUTPUT_DIRS} 
    ${cpg_include}
    ${HiGHS_INCLUDE_DIRS}
  )

  target_compile_options(py_mpcc PRIVATE -std=c++17 -O3)
  target_link_libraries(py_mpcc
    PRIVATE
    highs
    ${HPIPM_LIBRARIES}
    ${ACADOS_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    ${ACADOS_OUTPUT_LIBS}
    ${BLASFEO_LIBRARIES}
  )
else()
  message(STATUS "Pybind11 not found, mpcc python bindings will not be built")
endif()

########################
# ROS executables (only if catkin)
########################
if(catkin_FOUND)
  set(MPCC_SRCS
    src/mpcc/logger.cpp
    src/mpcc/mpcc_ros.cpp
    src/mpcc/mpcc_core.cpp
    src/mpcc/mpcc_acados.cpp
    src/mpcc/mpcc_di_acados.cpp
  )

  add_executable(mpcc_ros 
    ${MPCC_SRCS}
    src/mpcc/mpcc_node.cpp
  )
  target_link_libraries(mpcc_ros
    ${HPIPM_LIBRARIES}
    ${catkin_LIBRARIES}
    ${ACADOS_LIBRARIES}
    ${EIGEN3_LIBRARIES}
    ${ACADOS_OUTPUT_LIBS}
    ${BLASFEO_LIBRARIES}
    highs
  )
  add_dependencies(mpcc_ros ${PROJECT_NAME}_generate_messages ${ACADOS_TARGETS})
  target_compile_options(mpcc_ros PRIVATE -std=c++17 -O3)
  target_include_directories(mpcc_ros PRIVATE ${ACADOS_OUTPUT_DIRS} ${cpg_include} ${HiGHS_INCLUDE_DIRS})

  add_executable(publish_pf_pose src/utils/publish_pf_pose.cpp)
  target_link_libraries(publish_pf_pose ${catkin_LIBRARIES})

  add_executable(local_costmap src/utils/local_costmap.cpp)
  target_link_libraries(local_costmap ${catkin_LIBRARIES})
endif()

