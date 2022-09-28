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
        # u_k = HJB_policy(x_k, dt_plan, value_array, EoM, veh, sg, ag)

        # simulate forward one time step
        x_k1 = runge_kutta_4(x_k, u_k, dt_plan, EoM, veh, sg)    # TO-DO: needs K_sub for collision checking
        
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
    # building fast action list
    index, weight = interpolants(sg.state_grid, x_k)
    ia_complete = collect(1:length(ag.action_grid))
    ia_neighbor_srt_unq = unique(opt_ia_array[index[sortperm(weight, rev=true)]])
    ia_leftover_shuf_unq = shuffle(setdiff(ia_complete, opt_ia_array[index]))
    ia_fast_list = vcat(ia_neighbor_srt_unq, ia_leftover_shuf_unq)

    # assessing optimality
    value_k = interp_value(x_k, value_array, sg)
    epsilon = 0.75 * dt_plan

    value_k1_min = Inf
    ia_min = ag.action_grid[1]

    for ia in ia_fast_list
        if ia == 0
            continue
        end

        # simulates action one step forward
        x_k1 = runge_kutta_4(x_k, ag.action_grid[ia], dt_plan, EoM, veh, sg)
        value_k1 = interp_value(x_k1, value_array, sg)

        # checks if tested action passes near-optimal threshold
        if value_k1 < (value_k - epsilon)
            return ag.action_grid[ia]
        end

        # otherwise, stores best action found so far
        if value_k1 < value_k1_min
            value_k1_min = value_k1
            ia_min = ia
        end
    end

    return ag.action_grid[ia_min]
end