# solver.jl

# main function to iteratively calculate HJB value function
function solve_HJB_PDE(env, veh, EoM, sg, ag, dt_solve, Dv_tol, max_solve_steps)
    # initialize data arrays
    value_array, opt_ia_array, set_array = initialize_value_array(sg, env, veh)

    # main function loop
    Dv_max = Inf
    solve_step = 1
    gs_step = 1

    while Dv_max > Dv_tol && solve_step <= max_solve_steps
        Dv_max = 0.0
        for ind_m in sg.ind_gs_array[gs_step]
            x = sg.state_grid[ind_m...]
            ind_s = multi2single_ind(ind_m, sg)

            if set_array[ind_s] == 2
                v_kn1 = value_array[ind_s]
                
                value_array[ind_s], opt_ia_array[ind_s] = update_node_value(x, value_array, dt_solve, EoM, env, veh, sg, ag)
                
                v_k = value_array[ind_s]
                Dv = abs(v_k - v_kn1)

                if Dv > Dv_max
                    Dv_max = Dv
                end
            end
        end

        println("step: ", solve_step, ", Dv_max = ", Dv_max) #, "Dv_max = ", Dv_max)

        if gs_step == 2^dimensions(sg.state_grid)
            gs_step = 1
        else
            gs_step += 1
        end

        solve_step += 1
    end

    return value_array, opt_ia_array, set_array
end

function update_node_value(x, value_array, dt_solve, EoM, env, veh, sg, ag) 
    qval_min = Inf
    ia_opt_ijk = 1

    for ia in 1:length(ag.action_grid)
        a = ag.action_grid[ia]

        cost_p = get_cost(x, a, dt_solve)

        x_p = runge_kutta_4(x, a, dt_solve, EoM, veh, sg)
        val_p = interp_value(x_p, value_array, sg)

        qval_a = cost_p + val_p

        if qval_a < qval_min
            qval_min = qval_a
            ia_opt_ijk = ia
        end
    end

    val_ijk = qval_min
   
    return val_ijk, ia_opt_ijk
end

# (?): should this be moved out into definitions?
function get_cost(x_k, a_k, dt_solve)
    cost_k = dt_solve

    return cost_k
end

# initialize arrays
function initialize_value_array(sg, env, veh)
    value_array = zeros(Float64, length(sg.state_grid))
    opt_ia_array = zeros(Int, length(sg.state_grid))
    set_array = zeros(Int, length(sg.state_grid))

    for ind_m in sg.ind_gs_array[1]
        x = sg.state_grid[ind_m...]
        ind_s = multi2single_ind(ind_m, sg)

        if in_workspace(x, env, veh) == false || in_obstacle_set(x, env, veh) == true
            value_array[ind_s] = 1e5
            set_array[ind_s] = 0
        
        elseif in_target_set(x, env, veh) == true
            value_array[ind_s] = 0.0
            set_array[ind_s] = 1
        
        else
            value_array[ind_s] = 1e5
            set_array[ind_s] = 2
        end
    end

    return value_array, opt_ia_array, set_array
end