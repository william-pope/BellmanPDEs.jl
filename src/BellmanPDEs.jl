module BellmanPDEs

using StaticArrays
using GridInterpolations
using LazySets
using Random
using Plots

export VPolygon, VPolyCircle, SVector, SArray, Environment, VehicleBody, StateGrid,
    define_environment, define_vehicle, define_state_grid,
    solve_HJB_PDE, plan_path, 
    interp_value, in_target_set, in_obstacle_set, in_workspace,
    HJB_policy, approx_HJB_policy, reactive_policy, approx_reactive_policy,
    plot_HJB_value, plot_HJB_path, plot_path_value, state_to_body, state_to_body_circle

export discrete_time_EoM, propagate_state, interp_value, interpolate, optimize_action

include("definitions.jl")
include("solver.jl")
include("planner.jl")
include("utils.jl")
include("plotting.jl")

end