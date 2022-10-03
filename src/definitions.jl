# definitions.jl

struct Environment
    workspace::VPolygon
    obstacle_list::Array{Any}
    goal::VPolygon
end

struct Vehicle
    wheelbase::Float64
    body_dims::Array{Float64}
    origin_to_cent::Array{Float64}
    origin_body::VPolygon
end

struct StateGrid
    state_grid::RectangleGrid
    state_list_static::Array{Any}
    angle_wrap_array::Array{Bool}
    ind_gs_array::Array
end

struct ActionGrid
    action_grid::RectangleGrid
    action_list_static::Array{Any}
end

# defines environment geoemtry
function define_environment(workspace, obstacle_list, goal)
    env = Environment(workspace, obstacle_list, goal)
    return env
end

# defines vehicle geometry
function define_vehicle(wheelbase, body_dims, origin_to_cent)
    x0_min = origin_to_cent[1] - 1/2*body_dims[1]
    x0_max = origin_to_cent[1] + 1/2*body_dims[1]
    y0_min = origin_to_cent[2] - 1/2*body_dims[2]
    y0_max = origin_to_cent[2] + 1/2*body_dims[2]
    origin_body = VPolygon([[x0_min, y0_min], [x0_max, y0_min], [x0_max, y0_max], [x0_min, y0_max]])

    veh = Vehicle(wheelbase, body_dims, origin_to_cent, origin_body)
    return veh
end

# discretizes state space
function define_state_grid(state_space, dx_sizes, angle_wrap)
    state_iters = [minimum(axis):dx_sizes[i]:maximum(axis) for (i, axis) in enumerate(state_space)]
    state_grid = RectangleGrid(state_iters...)

    state_list_static = []
    for state in state_grid
        push!(state_list_static, SA[state...])
    end

    # Gauss-Seidel sweeping scheme
    gs_iters = [[0,1] for axis in state_space]
    gs_prod = Iterators.product(gs_iters...)
    gs_list = Iterators.map(tpl -> convert(SVector{length(gs_iters), Int}, tpl), gs_prod)

    # for sweep in gs_list, need to define ind_list
    ind_gs_array = []
    for (i_gs, gs) in enumerate(gs_list)

        # for axis in sweep = [0,1,1], reverse ind_iters
        ind_iters = Array{StepRange{Int64, Int64}}(undef, size(state_space,1))
        for (i_ax, ax) in enumerate(gs)
            if gs[i_ax] == 0.0
                # forward
                ind_iters[i_ax] = 1:1:size(state_iters[i_ax],1)
            else
                # reverse
                ind_iters[i_ax] = size(state_iters[i_ax],1):-1:1
            end
        end

        ind_prod = Iterators.product(ind_iters...)
        ind_list = Iterators.map(tpl -> convert(SVector{length(ind_iters), Int}, tpl), ind_prod)

        push!(ind_gs_array, ind_list)
    end

    sg = StateGrid(state_grid, state_list_static, angle_wrap, ind_gs_array)
    return sg
end

# discretizes action space
function define_action_grid(action_space, du_num_steps)
    action_iters = [range(minimum(axis), maximum(axis), du_num_steps[i]) for (i, axis) in enumerate(action_space)]
    action_grid = RectangleGrid(action_iters...)

    action_list_static = []
    for action in action_grid
        push!(action_list_static, SA[action...])
    end

    ag = ActionGrid(action_grid, action_list_static)
    return ag
end