module BellmanPDEs

using StaticArrays
using GridInterpolations
using LazySets
using Random
using Plots

export VPolygon, VPolyCircle, SVector, Environment, VehicleBody, StateGrid,
    define_environment, define_vehicle, define_state_grid,
    solve_HJB_PDE, plan_HJB_path, HJB_policy, rollout_policy
    plot_HJB_value, plot_HJB_path

export discrete_time_EoM, propagate_state, optimize_action

include("definitions.jl")
include("solver.jl")
include("planner.jl")
include("utils.jl")
include("plotting.jl")

end