# Helper to collect acados generated code
function(collect_acados_generated_code model_prefix)
  set(output_dir_c ${CMAKE_SOURCE_DIR}/scripts/${model}/c_generated_code)
  set(output_dir_cpp ${CMAKE_SOURCE_DIR}/scripts/${model}/cpp_generated_code)

  if(APPLE)
      set(lib_ext "dylib")
  else()
      set(lib_ext "so")
  endif()

  set(${model_prefix}_SRC
    ${output_dir_c}/acados_solver_${model}_mpcc.c
    ${output_dir_c}/acados_sim_solver_${model}_mpcc.c
    ${output_dir_c}/${model}_model/${model}_expl_ode_fun.c
    ${output_dir_c}/${model}_model/${model}_expl_vde_adj.c
    ${output_dir_c}/${model}_model/${model}_expl_vde_forw.c
    ${output_dir_c}/${model}_model/${model}_expl_ode_hess.c
    ${output_dir_c}/${model}_cost/${model}_cost_ext_cost_fun.c
    ${output_dir_c}/${model}_cost/${model}_cost_ext_cost_0_fun.c
    ${output_dir_c}/${model}_cost/${model}_cost_ext_cost_e_fun.c
    ${output_dir_c}/${model}_cost/${model}_cost_ext_cost_fun_jac.c
    ${output_dir_c}/${model}_cost/${model}_cost_ext_cost_0_fun_jac.c
    ${output_dir_c}/${model}_cost/${model}_cost_ext_cost_e_fun_jac.c
    ${output_dir_c}/${model}_cost/${model}_cost_ext_cost_fun_jac_hess.c
    ${output_dir_c}/${model}_cost/${model}_cost_ext_cost_0_fun_jac_hess.c
    ${output_dir_c}/${model}_cost/${model}_cost_ext_cost_e_fun_jac_hess.c
  )

  set(${model_prefix}_SRC_CPP
    ${output_dir_cpp}/compute_cbf_abv.cpp
    ${output_dir_cpp}/compute_lfh_abv.cpp
    ${output_dir_cpp}/compute_lgh_abv.cpp

    ${output_dir_cpp}/compute_cbf_blw.cpp
    ${output_dir_cpp}/compute_lfh_blw.cpp
    ${output_dir_cpp}/compute_lgh_blw.cpp
  )

  set(${model_prefix}_HEADERS
    ${output_dir_c}/acados_solver_${model}.h
    ${output_dir_c}/acados_sim_solver_${model}.h
    ${output_dir_c}/${model}_cost/${model}_cost.h
    ${output_dir_c}/${model}_model/${model}_model.h
  )

  set(${model_prefix}_LIBS
    ${output_dir_c}/libacados_ocp_solver_${model}.${lib_ext}
    ${output_dir_c}/libacados_sim_solver_${model}.${lib_ext}
  )

  set(${model_prefix}_OUTPUT_FILES 
    ${${model_prefix}_SRC_CPP} 
    ${${model_prefix}_SRC} 
    ${${model_prefix}_HEADERS}
    ${${model_prefix}_LIBS}
    PARENT_SCOPE
  )

  set(${model_prefix}_SRC "${${model_prefix}_SRC}" PARENT_SCOPE)
  set(${model_prefix}_HEADERS "${${model_prefix}_HEADERS}" PARENT_SCOPE)
  set(${model_prefix}_LIBS "${${model_prefix}_LIBS}" PARENT_SCOPE)

  set(${model_prefix}_SRC_CXX "${${model_prefix}_SRC_CPP}" PARENT_SCOPE)

endfunction()

