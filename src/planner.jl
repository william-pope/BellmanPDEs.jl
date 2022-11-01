# planner.jl

# main function to generate a path from an initial state to goal
function plan_path(x_0, policy::Function, get_actions::Function, get_cost::Function, Dt, q_value_array, value_array, env, veh, sg, max_plan_steps)
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
        a_k = policy(x_k, Dv_RC, get_actions, get_cost, Dt, q_value_array, value_array, veh, sg)

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

function HJB_policy(x_k, Dv_RC, get_actions::Function, get_cost::Function, Dt, q_value_array, value_array, veh, sg)
    actions, ia_set = get_actions(x_k, Dt, veh)

    _, _, ia_opt = optimize_action(x_k, ia_set, actions, get_cost, Dt, value_array, veh, sg)

    a_k_opt = actions[ia_opt]

    return a_k_opt
end

# NOTE: make more explicit that backup action is just the pure HJB policy
#   - removing RC actions is just a convenience step
function reactive_policy(x_k, Dv_RC, get_actions::Function, get_cost::Function, Dt, q_value_array, value_array, veh, sg)    
    # get actions for current state
    actions, ia_set = get_actions(x_k, Dt, veh)

    # A) find best phi for Dv given by reactive controller ---
    ia_RC_set = findall(a -> a[2] == Dv_RC, actions)
    _, val_RC_best, ia_RC_best = optimize_action(x_k, ia_RC_set, actions, get_cost, Dt, value_array, veh, sg)

    # check if [Dv_RC, phi_best_RC] is a valid action in static environment ---
    infty_set_lim = 50.0
    if val_RC_best <= infty_set_lim
        a_ro = actions[ia_RC_best]
 
        return a_ro
    end

    # B) if RC requested Dv is not valid, then find pure HJB best action ---
    ia_no_RC_set = filter(ia -> !(ia in ia_RC_set), ia_set)
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


#   - for given state, have 2^4=16 neighboring grid nodes surrounding it
#   - each neighboring grid node has a Q-value for every action (21 actions)
#   - RC will limit the action set to a single Dv, same limited set will be considered at each neighbor (7 actions)
#       - function: given Dv_RC, output limited ia set (ia_RC_set)
#   - at each neighbor, can get ia_RC_best by minimizing Q-values over limited RC set

#   - with ia_RC_best at each neighbor, can do same set operations from old approx method (sort, unique, ...)
#       - (?): need to assemble list, or just try nearest neighbor first?
#   - will end up with short list of actions
#       - these are the ia_RC_best actions at the exact neighboring node states
#       - in other words, would be true optimal action if vehicle was at node state
#       - instead, can just assume that closest node action is near optimal for our state
#       - need to check subsequent state for collisions/RIC (value-based), but otherwise can take action

#   - (?): is value being bounded? or just checking validity
#       - bounding value is a little trickier than before, because RC constraint means none of the RC neighbors may be true optimal to Dt cost function
#       - however, minimum RC Q-value at neighboring nodes is true best case performance under RC constraint, should be able to use this if needed

#   - MAIN IDEA: instead of perfoming Q-value optimization at exact state, choose minimum from stored Q-values at neighboring nodes


function approx_reactive_policy(x_k, Dv_RC, get_actions::Function, get_cost::Function, Dt, q_value_array, value_array, veh, sg)
    # get actions for current state
    actions, ia_set = get_actions(x_k, Dt, veh)

    # A) get near-optimal RC action from nearest neighbor ---
    # limit action set based on reactive controllers
    ia_RC_set = findall(a -> a[2] == Dv_RC, actions)
    
    # find nearest neighbor in state grid
    ind_s_nbrs, weights_nbrs = interpolants(sg.state_grid, x_k)
    ind_s_NN = ind_s_nbrs[findmax(weights_nbrs)[2]]

    # minimize Q-value over RC limited action set to find best action
    qvals_NN = q_value_array[ind_s_NN]
    ia_NN_RC_best = argmin(ia -> qvals_NN[ia], ia_RC_set)

    # check if ia_NN_RC_best is a valid action
    x_p, _ = propagate_state(x_k, actions[ia_NN_RC_best], Dt, veh)
    val_NN_RC_best = interp_value(x_p, value_array, sg)

    infty_set_lim = 50.0
    if val_NN_RC_best <= infty_set_lim
        a_ro = actions[ia_NN_RC_best]

        return a_ro
    end

    # B) if NN RC action is not valid, then find pure HJB best action ---
    _, _, ia_opt = optimize_action(x_k, ia_set, actions, get_cost, Dt, value_array, veh, sg)

    a_ro = actions[ia_opt]
    
    return a_ro
end