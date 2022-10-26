# planner.jl

# main function to generate a path from an initial state to goal
function plan_HJB_path(x_0, dt_plan, value_array, opt_ia_array, max_steps, EoM, env, veh, sg, ag)
    x_k = x_0

    x_path = []
    u_path = []

    step = 1
    while in_target_set(x_k, env, veh) == false && step < max_steps
        # calculate optimal action
        u_k = fast_policy(x_k, dt_plan, value_array, opt_ia_array, EoM, veh, sg, ag)

        # simulate forward one time step
        x_k1 = runge_kutta_4(x_k, u_k, dt_plan, EoM, veh, sg)

        # store state and action at current time step
        push!(x_path, x_k)
        push!(u_path, u_k)

        # pass state forward to next step
        x_k = deepcopy(x_k1)
        step += 1
    end

    push!(x_path, x_k)

    return x_path, u_path, step
end

# calculate one-step lookahead search at current state
function HJB_policy(x_k, dt_plan, value_array, EoM, veh, sg, ag)
    value_k1_min = Inf
    a_k_opt = ag.action_grid[1]

    for a_k in ag.action_grid
        x_k1 = runge_kutta_4(x_k, a_k, dt_plan, EoM, veh, sg)
        value_k1 = interp_value(x_k1, value_array, sg)

        if value_k1 < value_k1_min
            value_k1_min = value_k1
            a_k_opt = a_k
        end
    end

    return a_k_opt
end

# use stored action grid to efficiently find near-optimal action at current state
function fast_policy(x_k, dt_plan, value_array, opt_ia_array, EoM, veh, sg, ag)
    # gets actions from neighboring nodes
    nbr_indices, nbr_weights = interpolants(sg.state_grid, x_k)
    coord_srt = sortperm(nbr_weights, rev=true)
    nbr_indices_srt = view(nbr_indices, coord_srt)
    ia_neighbor_srt_unq = opt_ia_array[nbr_indices]
    unique!(ia_neighbor_srt_unq)

    # assesses optimality
    value_k = interp_value(x_k, value_array, sg)
    epsilon = 0.75 * dt_plan

    ia_min = 1
    value_k1_min = Inf

    # checks neighbors first
    for ia in ia_neighbor_srt_unq
        if ia == 0
            continue
        end

        # simulates action one step forward
        x_k1 = runge_kutta_4(x_k, ag.action_list_static[ia], dt_plan, EoM, veh, sg)
        value_k1 = interp_value(x_k1, value_array, sg)

        # checks if tested action passes near-optimal threshold
        if value_k1 < (value_k - epsilon)
            return ag.action_list_static[ia]
        end

        # otherwise, stores best action found so far
        if value_k1 < value_k1_min
            value_k1_min = value_k1
            ia_min = ia
        end
    end

    # if optimal threshold hasn't been met (return), continues to check rest of actions
    ia_complete = collect(1:length(ag.action_grid))
    ia_leftover_shuf_unq = shuffle(setdiff(ia_complete, opt_ia_array[nbr_indices]))

    for ia in ia_leftover_shuf_unq
        if ia == 0
            continue
        end

        # simulates action one step forward
        x_k1 = runge_kutta_4(x_k, ag.action_list_static[ia], dt_plan, EoM, veh, sg)
        value_k1 = interp_value(x_k1, value_array, sg)

        # checks if tested action passes near-optimal threshold
        if value_k1 < (value_k - epsilon)
            return ag.action_list_static[ia]
        end

        # otherwise, stores best action found so far
        if value_k1 < value_k1_min
            value_k1_min = value_k1
            ia_min = ia
        end
    end

    return ag.action_list_static[ia_min]
end
