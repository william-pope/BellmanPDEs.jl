# planner.jl

# main function to generate a path from an initial state to goal
function plan_HJB_path(x_0, get_actions::Function, get_cost::Function, Dt, q_value_array, value_array, env, veh, sg, max_plan_steps)
    val_0 = interp_value(x_0, value_array, sg)
    
    x_path = []
    x_subpath = []  
    a_path = []
    val_path = []

    push!(x_path, x_0)
    push!(x_subpath, x_0)
    push!(val_path, val_0)

    x_k = x_0
   
    for plan_step in 1:max_plan_steps
        # calculate optimal action
        a_k = HJB_policy(x_k, get_actions, get_cost, Dt, value_array, veh, sg)

        # simulate forward one time step
        x_k1, x_k1_subpath = propagate_state(x_k, a_k, Dt, veh)

        # take value at current state (for plotting)
        val_k1 = interp_value(x_k1, value_array, sg)
        
        # store state and action at current time step
        push!(x_path, x_k1)
        for x_kk in x_k1_subpath
            push!(x_subpath, x_kk)
        end
        push!(a_path, a_k)
        push!(val_path, val_k1)

        # check if termination condition met
        if in_target_set(x_k1, env, veh) == true
            break
        end
        
        # pass state forward to next step
        x_k = deepcopy(x_k1)
    end

    return x_path, x_subpath, a_path, val_path
end

function plan_rollout_path(x_0, get_actions::Function, get_cost::Function, Dt, q_value_array, value_array, env, veh, sg, max_plan_steps)
    val_0 = interp_value(x_0, value_array, sg)
    
    x_path = []
    x_subpath = []  
    a_path = []
    val_path = []

    push!(x_path, x_0)
    push!(x_subpath, x_0)
    push!(val_path, val_0)

    x_k = x_0
   
    for plan_step in 1:max_plan_steps
        # calculate rollout action
        Dv_RC = rand([-0.5, 0.0, 0.5])
        a_k = rollout_policy(x_k, Dv_RC, get_actions, get_cost, Dt, value_array, veh, sg)

        # simulate forward one time step
        x_k1, x_k1_subpath = propagate_state(x_k, a_k, Dt, veh)

        # take value at current state (for plotting)
        val_k1 = interp_value(x_k1, value_array, sg)
        
        # store state and action at current time step
        push!(x_path, x_k1)
        for x_kk in x_k1_subpath
            push!(x_subpath, x_kk)
        end
        push!(a_path, a_k)
        push!(val_path, val_k1)

        # check if termination condition met
        if in_target_set(x_k1, env, veh) == true
            break
        end
        
        # pass state forward to next step
        x_k = deepcopy(x_k1)
    end

    return x_path, x_subpath, a_path, val_path
end

function HJB_policy(x_k, get_actions::Function, get_cost::Function, Dt, value_array, veh, sg)
    actions = get_actions(x_k, Dt, veh)
    ia_set = collect(1:length(actions))     # SPEED: should be a faster way to do this

    _, _, ia_opt = optimize_action(x_k, ia_set, actions, get_cost, Dt, value_array, veh, sg)

    a_k_opt = actions[ia_opt]

    return a_k_opt
end

function rollout_policy(x_k, Dv_RC, get_actions::Function, get_cost::Function, Dt, value_array, veh, sg)    
    # get actions for current state
    actions = get_actions(x_k, Dt, veh)

    # 1) find best phi for Dv given by reactive controller ---
    ia_RC_set = findall(Dv -> Dv == Dv_RC, getindex.(actions, 2))   # SPEED: might be a faster way to do this
    _, val_RC_best, ia_RC_best = optimize_action(x_k, ia_RC_set, actions, get_cost, Dt, value_array, veh, sg)

    # 2) check if [Dv_RC, phi_best_RC] is a valid action in static environment ---
    infty_set_lim = 50.0
    if val_RC_best <= infty_set_lim
        a_ro = actions[ia_RC_best]
 
        return a_ro
    end

    # 3) if RC requested Dv is not valid, then find pure HJB best action ---
    ia_no_RC_set = findall(Dv -> Dv != Dv_RC, getindex.(actions, 2))
    _, _, ia_no_RC_best = optimize_action(x_k, ia_no_RC_set, actions, get_cost, Dt, value_array, veh, sg)

    a_ro = actions[ia_no_RC_best]

    return a_ro
end

# approx reactive HJB_policy
#   - need to store RC best action for each possible Dv input {-, 0, +}
#   - create multi-dimensional version of ia_opt_array, should be 3 ia_RC_best stored
#   - don't want to modify optimize_actions(), just feed in different ia_sets for each Dv
#       - want to define these sets once, so don't have to keep using findall(Dv)

#   - could also just store Q(s,a) for every action at every state
#   - would do Dv_RC filtering during application, instead of having to deal with specifically during solving
#   - for each state, would have array of values same length as action set
#   - would just need to do some find/min operations on the list to find desired action indices

function approx_rollout_policy(x_k, Dv_RC, get_actions::Function, get_cost::Function, Dt, q_value_array, value_array, veh, sg)
    # get actions for current state
    actions_RO = get_actions(x_k, Dt, veh)

    # get nearest neighbor phi set
    @show grid_indices_nbr, dist_weights_nbr = interpolants(sg.state_grid, x_k)
    @show coord_srt = sortperm(dist_weights_nbr, rev=true)
    @show grid_indices_nbr_srt = view(grid_indices_nbr, coord_srt)
    @show ia_nbr_srt_unq = ia_opt_array[grid_indices_nbr_srt]
    @show unique!(ia_nbr_srt_unq)

    actions_NN = actions_RO[ia_nbr_srt_unq]

    display(actions_NN)

    return 1

    # # 1) [phi_NN, Dv_RC]
    # ia_NN_RC_set = 

    # # 2) [phi_no_NN, Dv_RC]
    # ia_nNN_RC_set = 

    # # 3) [phi_NN, Dv_no_RC]
    # ia_NN_nRC_set = 

    # # 4) [phi_no_NN, Dv_no_RC]
    # ia_nNN_nRC_set = 

    return a_ro
end