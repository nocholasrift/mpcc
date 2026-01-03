#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <iostream>

#include <mpcc/map_util.h>
#include <mpcc/mpcc_core.h>

#include <mpcc/tube_gen.h>

namespace py = pybind11;
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::VectorXd>);

PYBIND11_MODULE(py_mpcc, m) {
  py::bind_vector<std::vector<Eigen::VectorXd>>(m, "vec_VecXd");

  py::enum_<MPCType>(m, "MPCType")
      .value("DOUBLE_INTEGRATOR", MPCType::kDoubleIntegrator)
      .value("UNICYCLE", MPCType::kUnicycle);

  py::class_<MPCCore>(m, "MPCCore")
      .def(py::init<>())
      .def(py::init<const MPCType&>())
      .def("load_params", &MPCCore::load_params)
      .def("get_params", &MPCCore::get_params)
      .def("set_odom", &MPCCore::set_odom)
      .def("set_tubes", &MPCCore::set_tubes)
      .def("set_trajectory",
           (void (MPCCore::*)(const Eigen::VectorXd&, const Eigen::VectorXd&,
                              int,
                              const Eigen::VectorXd&))&MPCCore::set_trajectory)
      .def("get_s_from_pose", &MPCCore::get_s_from_pose)
      .def("get_cbf_data", &MPCCore::get_cbf_data)
      .def("get_horizon", &MPCCore::get_horizon)
      .def("solve", &MPCCore::solve)
      .def("get_mpc_command", &MPCCore::get_mpc_command)
      .def("get_solver_status", &MPCCore::get_solver_status)
      .def("get_cbf_data", &MPCCore::get_cbf_data)
      .def("get_input_limits", &MPCCore::get_input_limits)
      .def("get_state_limits", &MPCCore::get_state_limits)
      .def("get_state", &MPCCore::get_state);

  py::class_<map_util::OccupancyGrid>(m, "OccupancyGrid", py::module_local())
      .def(py::init<>())
      .def(py::init<int, int, double, double, double,
                    std::vector<unsigned char>&,
                    const std::vector<unsigned char>&,
                    const std::vector<unsigned char>&>())
      .def("get_origin", &map_util::OccupancyGrid::get_origin)
      .def("world_to_map", &map_util::OccupancyGrid::world_to_map)
      .def("cells_to_index", &map_util::OccupancyGrid::cells_to_index)
      .def("get_cost", py::overload_cast<unsigned int, const std::string&>(
                           &map_util::OccupancyGrid::get_cost))
      .def("update", py::overload_cast<int, int, double, double, double,
                                       const std::vector<unsigned char>&,
                                       const std::vector<unsigned char>&,
                                       const std::vector<unsigned char>&>(
                         &map_util::OccupancyGrid::update));

  // tube generation
#ifndef FOUND_CATKIN
  m.def("get_tubes", &tube_utils::get_tubes2);
#endif
}
