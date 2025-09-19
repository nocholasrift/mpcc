#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mpcc/mpcc_core.h"

namespace py = pybind11;

PYBIND11_MODULE(py_mpcc, m) {

  py::class_<MPCCore>(m, "MPCCore")
      .def(py::init<>())
      .def(py::init<const std::string&>())
      .def("load_params", &MPCCore::load_params)
      .def("get_params", &MPCCore::get_params)
      .def("set_odom", &MPCCore::set_odom)
      .def("set_tubes", &MPCCore::set_tubes)
      .def("set_trajectory",
           (void(MPCCore::*)(const Eigen::VectorXd&, const Eigen::VectorXd&,
                             int, const Eigen::VectorXd&)) &
               MPCCore::set_trajectory)
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
}
