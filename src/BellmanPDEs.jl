module BellmanPDEs

using StaticArrays
using GridInterpolations
using LazySets
using Random

export VPolygon, VPolyCircle, SVector, Environment, VehicleBody, StateGrid, ActionGrid,
    define_environment, define_vehicle, define_state_grid, define_action_grid,
    solve_HJB_PDE, plan_HJB_path, fast_policy

include("definitions.jl")
include("solver.jl")
include("planner.jl")
include("utils.jl")

end
