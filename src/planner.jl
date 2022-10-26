# planner.jl

# main function to generate a path from an initial state to goal
function plan_HJB_path(x_0, get_actions::Function, get_cost::Function, Dt, value_array, a_ind_opt_array, env, veh, sg, max_plan_steps)
    x_path = []
    x_subpath = []  
    a_path = []

    x_k = x_0
    push!(x_path, x_k)
    push!(x_subpath, x_k)
   
    for plan_step in 1:max_plan_steps
        # calculate optimal action
        a_k = HJB_policy(x_k, get_actions, get_cost, Dt, value_array, veh, sg)

        # simulate forward one time step
        x_k1, x_k1_subpath = propagate_state(x_k, a_k, Dt, veh)
        
        # store state and action at current time step
        push!(x_path, x_k1)
        for x_kk in x_k1_subpath
            push!(x_subpath, x_kk)
        end
        push!(a_path, a_k)

        # check if termination condition met
        if in_target_set(x_k1, env, veh) == true
            break
        end
        
        # pass state forward to next step
        x_k = deepcopy(x_k1)
    end

    return x_path, x_subpath, a_path
end

function HJB_policy(x_k, get_actions::Function, get_cost::Function, Dt, value_array, veh, sg)
    ro_actions = get_actions(x_k, Dt, veh)
    a_ind_array = collect(1:length(ro_actions))

    a_ind_opt, _ = optimize_action(x_k, a_ind_array, ro_actions, get_cost, Dt, value_array, veh, sg)

    a_k_opt = ro_actions[a_ind_opt]

    return a_k_opt
end

function rollout_policy(x_k, Dv_RC, get_actions::Function, get_cost::Function, Dt, value_array, veh, sg)    
    # get actions for current state
    ro_actions = get_actions(x_k, Dt, veh)

    # 1) find best phi for Dv given by reactive controller ---
    a_ind_array_RC = findall(Dv -> Dv == Dv_RC, getindex.(ro_actions, 2))
    a_ind_best_RC, val_best_RC = optimize_action(x_k, a_ind_array_RC, ro_actions, get_cost, Dt, value_array, veh, sg)

    # 2) check if [Dv_RC, phi_best_RC] is a valid action in static environment ---
    infty_set_lim = 50.0
    if val_best_RC <= infty_set_lim
        a_ro = ro_actions[a_ind_best_RC]
 
        return a_ro
    end

    # 3) if RC requested Dv is not valid, then find pure HJB best action ---
    a_ind_array_no_RC = findall(Dv -> Dv != Dv_RC, getindex.(ro_actions, 2))
    a_ind_best_HJB, _ = optimize_action(x_k, a_ind_array_no_RC, ro_actions, get_cost, Dt, value_array, veh, sg)

    a_ro = ro_actions[a_ind_best_HJB]

    return a_ro
end
