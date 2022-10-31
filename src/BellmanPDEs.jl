module BellmanPDEs

using StaticArrays
using GridInterpolations
using LazySets
using Random
using Plots

export VPolygon, VPolyCircle, SVector, SArray, Environment, VehicleBody, StateGrid,
    define_environment, define_vehicle, define_state_grid,
    solve_HJB_PDE, plan_HJB_path, plan_rollout_path, HJB_policy, rollout_policy, approx_rollout_policy,
    plot_HJB_value, plot_HJB_path, plot_path_value

export discrete_time_EoM, propagate_state, interp_value, interpolate, optimize_action

include("definitions.jl")
include("solver.jl")
include("planner.jl")
include("utils.jl")
include("plotting.jl")

end