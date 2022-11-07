# solver.jl

# main function to iteratively calculate HJB value function
function solve_HJB_PDE(get_actions::Function, get_reward::Function, Dt, env, veh, sg, Dval_tol, max_solve_steps)
    # initialize data arrays
    q_value_array, value_array, set_array = initialize_value_array(Dt, get_actions, sg, env, veh)

    num_gs_sweeps = 2^dimensions(sg.state_grid)

    # main function loop
    gs_step = 1

    for solve_step in 1:max_solve_steps
        Dval_max = 0.0

        for ind_m in sg.ind_gs_array[gs_step]
            ind_s = multi2single_ind(ind_m, sg)     # SPEED: able to speed up? haven't looked

            # if the node is in free space, update its value
            if set_array[ind_s] == 2
                x = sg.state_list_static[ind_s]     # SPEED: list might be better as an MVector (actually big vector, so actually not?)
                
                # store previous value
                v_kn1 = value_array[ind_s]
                
                # calculate new value
                q_value_array[ind_s], value_array[ind_s], _ = update_node_value(x, get_actions, get_reward, Dt, value_array, veh, sg)
                
                # compare old and new values, update largest change in value
                v_k = value_array[ind_s]
                Dval = abs(v_k - v_kn1)

                if Dval > Dval_max
                    Dval_max = Dval
                end
            end
        end

        println("solve_step: ", solve_step, ", gs_step: ", gs_step, ", Dval_max = ", Dval_max)

        # check if termination condition met
        if Dval_max <= Dval_tol
            break
        end

        # update Gauss-Seidel counter
        if gs_step == num_gs_sweeps
            gs_step = 1
        else
            gs_step += 1
        end
    end

    return q_value_array, value_array
end

# ISSUE: seems like q_value_array is not being updated properly
function update_node_value(x, get_actions::Function, get_reward::Function, Dt, value_array, veh, sg) 
    # using entire action set
    actions, ia_set = get_actions(x, Dt, veh)

    # find optimal action and value at state
    qval_x_array, val_x, ia_opt = optimize_action(x, ia_set, actions, get_reward::Function, Dt, value_array, veh, sg)
   
    return qval_x_array, val_x, ia_opt
end

# initialize arrays
function initialize_value_array(Dt, get_actions::Function, sg, env, veh)
    x = sg.state_list_static[1]
    _, ia_set = get_actions(x, Dt, veh)
    
    q_value_array = Vector{Vector{Float64}}(undef, length(sg.state_grid))
    value_array = Vector{Float64}(undef, length(sg.state_grid))
    set_array = Vector{Int}(undef, length(sg.state_grid))

    for ind_m in sg.ind_gs_array[1]
        ind_s = multi2single_ind(ind_m, sg)
        x = sg.state_list_static[ind_s]

        if in_workspace(x, env, veh) == false || in_obstacle_set(x, env, veh) == true
            q_value_array[ind_s] = -1e6 * ones(length(ia_set))
            value_array[ind_s] = -1e6
            set_array[ind_s] = 0
        
        elseif in_target_set(x, env, veh) == true
            q_value_array[ind_s] = 1000.0 * ones(length(ia_set))
            value_array[ind_s] = 1000.0
            set_array[ind_s] = 1
        
        else
            q_value_array[ind_s] = -1e6 * ones(length(ia_set))
            value_array[ind_s] = -1e6
            set_array[ind_s] = 2
        end
    end

    return q_value_array, value_array, set_array
end