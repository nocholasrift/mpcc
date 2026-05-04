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
  # amrl_logging
)

# find_package(pybind11 QUIET)
find_package(Python3 REQUIRED)
find_package(ACADOS REQUIRED)
find_package(BLASFEO REQUIRED)
find_package(HPIPM REQUIRED)
find_package(Eigen3 REQUIRED)

add_subdirectory(third_party/HiGHS)
add_subdirectory(third_party/pybind11)


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

# set(ACADOS_INCLUDE_DIRS
#   third_party/acados/include
#   third_party/acados/interfaces/acados_c
#   third_party/acados/include/blasfeo/include
#   third_party/acados/include/hpipm/include
# )
# message(STATUS "ACADOS_INCLUDE_DIRS: ${ACADOS_INCLUDE_DIRS}")

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

set(CASADI_OUTPUT_LIBS "")

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
    ${ACADOS_GEN_DIR}
    ${CASADI_GEN_DIR}
  )

  collect_acados_generated_code(${model} ${ACADOS_GEN_DIR} ${CASADI_GEN_DIR})
  list(APPEND ACADOS_OUTPUT_SRC_CXX ${${model}_SRC_CXX})
  list(APPEND ACADOS_OUTPUT_HEADERS ${${model}_HEADERS})
  list(APPEND ACADOS_OUTPUT_HEADERS_CXX ${${model}_HEADERS_CXX})
  list(APPEND ACADOS_OUTPUT_LIBS ${${model}_LIBS})
  list(APPEND CASADI_OUTPUT_LIBS ${${model}_CASADI_LIBS})

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

########################
# Acados Static Library
########################
add_library(acados_generated_lib STATIC 
    ${ACADOS_OUTPUT_SRC_CXX}
)

# Apply includes to the generated library
target_include_directories(acados_generated_lib PUBLIC 
    ${ACADOS_OUTPUT_DIRS}
    ${CASADI_GEN_DIR}
    ${ACADOS_INCLUDE_DIRS}
    ${BLASFEO_INCLUDE_DIRS}
    ${HPIPM_INCLUDE_DIRS}
)
add_dependencies(acados_generated_lib ${ACADOS_TARGETS})
target_link_libraries(acados_generated_lib 
    ${ACADOS_LIBRARIES}
    ${HPIPM_LIBRARIES}
    ${BLASFEO_LIBRARIES}
    ${ACADOS_OUTPUT_LIBS}
    # ${ACADOS_OUTPUT_LIBS}
    # ${CASADI_OUTPUT_LIBS}
)
target_compile_options(acados_generated_lib PRIVATE -fPIC)


########################
# MPCC Core Library
########################
add_library(mpcc_core SHARED
  src/mpcc/common/mpcc_core.cpp
  src/mpcc/unicycle/mpcc_unicycle_acados.cpp
  src/mpcc/unicycle/casadi_unicycle_interface.cpp
  src/mpcc/double_integrator/mpcc_di_acados.cpp
  src/mpcc/double_integrator/casadi_double_integrator_interface.cpp
)
target_compile_options(mpcc_core PRIVATE -std=c++17 -O2 -fPIC)
target_include_directories(mpcc_core PRIVATE 
  ${CASADI_GEN_DIR}
  ${ACADOS_OUTPUT_DIRS} 
  ${HiGHS_INCLUDE_DIRS}
)

target_link_libraries(mpcc_core
  ${HPIPM_LIBRARIES}
  ${ACADOS_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${ACADOS_OUTPUT_LIBS}
  ${CASADI_OUTPUT_LIBS}
  ${BLASFEO_LIBRARIES}
  acados_generated_lib
  highs
)

########################
# Python bindings (optional)
########################
option(BUILD_PYTHON_BINDINGS "Build pybind11 Python bindings" ON)

if(BUILD_PYTHON_BINDINGS)
  find_package(Python3 COMPONENTS Development QUIET)

  if(Python3_Development_FOUND AND TARGET pybind11::module)
    message(STATUS "Pybind11 v${pybind11_VERSION} found, building mpcc python bindings")
    add_definitions(-DFOUND_PYBIND11)

    pybind11_add_module(py_mpcc
      src/mpcc/py_mpcc.cpp
    )

    target_include_directories(py_mpcc PRIVATE
      ${ACADOS_OUTPUT_DIRS}
      ${HiGHS_INCLUDE_DIRS}
      ${Python3_INCLUDE_DIRS}
    )

    target_compile_options(py_mpcc PRIVATE -std=c++17 -O2)
    target_link_libraries(py_mpcc
      PRIVATE
      mpcc_core
      ${CASADI_OUTPUT_LIBS}
    )
  else()
    message(STATUS "Pybind11 or Python3 development headers not found, skipping Python bindings")
  endif()
else()
  message(STATUS "BUILD_PYTHON_BINDINGS=OFF, skipping Python bindings")
endif()


########################
# ROS executables (only if catkin)
########################
if (DEFINED ENV{ROS_VERSION})
  if ($ENV{ROS_VERSION} EQUAL 1)
    set(ROS1 TRUE)
  elseif ($ENV{ROS_VERSION} EQUAL 2)
    set(ROS2 TRUE)
  endif()
else()
  find_package(catkin QUIET)
  if (catkin_FOUND)
    set(ROS1 TRUE)
  else()
    find_package(ament_cmake QUIET)
    if (ament_cmake_FOUND)
      set(ROS2 TRUE)
    endif()
  endif()
endif()

if (ROS1 OR ROS2)
  set(MSG_FILES "msg/RLState.msg")
  set(SRV_FILES "srv/QuerySAC.srv" "srv/QuerySACDI.srv")
endif()

if (ROS1)
  message(STATUS "Catkin found, building ROS1 nodes")

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

  add_executable(mpcc_ros 
    mpcc_ros1/src/logger.cpp
    mpcc_ros1/src/mpcc_ros.cpp
    mpcc_ros1/src/mpcc_node.cpp
  )

  target_link_libraries(mpcc_ros
    ${catkin_LIBRARIES}
    mpcc_core
  )

  # add_dependencies(mpcc_ros ${PROJECT_NAME}_generate_messages ${ACADOS_TARGETS})
  target_compile_options(mpcc_ros PRIVATE -std=c++17 -O2)
  target_include_directories(mpcc_ros PRIVATE 
    ${ACADOS_OUTPUT_DIRS} 
    ${HiGHS_INCLUDE_DIRS}
    mpcc_ros1/include
  )

  add_executable(publish_pf_pose mpcc_ros1/src/utils/publish_pf_pose.cpp)
  target_link_libraries(publish_pf_pose ${catkin_LIBRARIES})

  # add_executable(local_costmap src/utils/local_costmap.cpp)
  # target_link_libraries(local_costmap ${catkin_LIBRARIES})
elseif(ROS2)
  message(STATUS "Catkin found, building ROS2 nodes")
  find_package(tf2 REQUIRED)
  find_package(rclcpp REQUIRED)
  find_package(std_msgs REQUIRED)
  find_package(nav_msgs REQUIRED)
  find_package(std_srvs REQUIRED)
  find_package(sensor_msgs REQUIRED)
  find_package(ament_cmake REQUIRED)
  find_package(geometry_msgs REQUIRED)
  find_package(trajectory_msgs REQUIRED)
  find_package(nav2_costmap_2d REQUIRED)
  find_package(tf2_geometry_msgs REQUIRED)
  find_package(rosidl_default_generators REQUIRED)

  rosidl_generate_interfaces(${PROJECT_NAME}
    ${MSG_FILES}
    ${SRV_FILES}
    DEPENDENCIES geometry_msgs nav_msgs trajectory_msgs
  )

  add_executable(mpcc_ros
    mpcc_ros2/src/logger_ros2.cpp
    mpcc_ros2/src/mpcc_ros2.cpp
    mpcc_ros2/src/mpcc_node2.cpp
  )
  target_compile_options(mpcc_ros PRIVATE -std=c++17 -O2)

  rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")
  target_link_libraries(mpcc_ros
    mpcc_core
    # tf2::tf2
    # tf2_geometry_msgs::tf2_geometry_msgs
    ${cpp_typesupport_target}
  )

  ament_target_dependencies(mpcc_ros
    rclcpp
    trajectory_msgs
    nav2_costmap_2d
  )

  target_include_directories(mpcc_ros
    PRIVATE
    mpcc_ros2/include
    ${ACADOS_OUTPUT_DIRS}
    ${HiGHS_INCLUDE_DIRS}
  )


  install(TARGETS mpcc_ros
    DESTINATION lib/${PROJECT_NAME}
  )

  add_executable(publish_pf_pose mpcc_ros2/src/utils/publish_pf_pose.cpp)
  ament_target_dependencies(publish_pf_pose
    rclcpp
    nav_msgs
    tf2
    tf2_ros
    tf2_geometry_msgs
  )
  install(TARGETS publish_pf_pose DESTINATION lib/${PROJECT_NAME})

  ament_export_dependencies(rosidl_default_runtime)

  install(DIRECTORY
    mpcc_ros2/launch
    mpcc_ros2/params
    # params
    DESTINATION share/${PROJECT_NAME}
  )

  # install python scripts
  install(DIRECTORY
    mpcc_ros2/scripts/
    DESTINATION lib/${PROJECT_NAME}
  )

  ament_package()
else()
  message(STATUS "Ros not found, skipping")
endif()
