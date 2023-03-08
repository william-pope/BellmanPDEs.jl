# utils.jl

# NOTE: needs to be general for:
#   - solving value grid
#   - planning over full action space
#   - planning over smaller action space

#   - val_x - best value result out of actions tested (single Float64)
#   - ia_opt - action tested that produced best value (single Int)
#   - qval_x_array - value produced by each of actions tested (array of Float64, length(ia_set))

function optimize_action(x, ia_set, actions, get_reward::Function, Dt, value_array, veh, sg)    
    qval_x_array = zeros(Float64, length(ia_set))  
    
    # iterate through all given action indices
    for ja in eachindex(ia_set)
        a = actions[ia_set[ja]]

        reward_x_a = get_reward(x, a, Dt, veh)

        x_p, _ = propagate_state(x, a, Dt, veh)
        val_xp = interp_value(x_p, value_array, sg)

        qval_x_array[ja] = reward_x_a + val_xp
    end

    # get value
    val_x = maximum(qval_x_array)

    # get optimal action index
    ja_opt = findmax(qval_x_array)[2]
    ia_opt = ia_set[ja_opt]

    return qval_x_array, val_x, ia_opt
end

function propagate_state(x_k, a_k, Dt, veh)
    # define number of substeps used for integration
    substeps = 10
    Dt_sub = Dt / substeps

    x_k1_subpath = MVector{substeps, SVector{4, Float64}}(undef)

    # step through substeps from x_k
    x_kk = x_k
    for kk in 1:substeps
        # Dv applied on first substep only
        kk == 1 ? a_kk = a_k : a_kk = SVector{2, Float64}(a_k[1], 0.0)
            
        # propagate for Dt_sub
        x_kk1 = discrete_time_EoM(x_kk, a_kk, Dt_sub, veh)

        # store new point in subpath
        x_k1_subpath[kk] = x_kk1

        # pass state to next loop
        x_kk = x_kk1
    end

    x_k1 = x_kk

    return x_k1, x_k1_subpath
end

function discrete_time_EoM(x_k, a_k, Dt, veh)
    # break out current state
    xp_k = x_k[1]
    yp_k = x_k[2]
    theta_k = x_k[3]
    v_k = x_k[4]

    # break out action
    phi_k = a_k[1]
    Dv_k = a_k[2]

    # calculate derivative at current state
    xp_dot_k = (v_k + Dv_k) * cos(theta_k)
    yp_dot_k = (v_k + Dv_k) * sin(theta_k)
    theta_dot_k = (v_k + Dv_k) * 1/veh.l * tan(phi_k)

    # calculate next state with linear approximation
    xp_k1 = xp_k + (xp_dot_k * Dt)
    yp_k1 = yp_k + (yp_dot_k * Dt)
    theta_k1 = theta_k + (theta_dot_k * Dt)
    v_k1 = v_k + (Dv_k)

    # modify angle to be within [-pi, +pi] range
    theta_k1 = theta_k1 % (2*pi)
    if theta_k1 > pi
        theta_k1 -= 2*pi
    elseif theta_k1 < -pi
        theta_k1 += 2*pi
    end

    # reassemble state vector
    x_k1 = SVector{4, Float64}(xp_k1, yp_k1, theta_k1, v_k1)

    return x_k1
end

function interp_value(x, value_array, sg)
    # check if current state is within state space
    for d in eachindex(x)
        if x[d] < sg.state_grid.cutPoints[d][1] || x[d] > sg.state_grid.cutPoints[d][end]
            val_x = -1e6

            return val_x
        end
    end

    # interpolate value at given state
    val_x = interpolate(sg.state_grid, value_array, x)

    return val_x
end

function interp_value_NN(x, value_array, sg)
    # check if current state is within state space
    for d in eachindex(x)
        if x[d] < sg.state_grid.cutPoints[d][1] || x[d] > sg.state_grid.cutPoints[d][end]
            val_x = -1e6

            return val_x
        end
    end

    # take nearest-neighbor value
    ind_s_nbrs, weights_nbrs = interpolants(sg.state_grid, x)
    ind_s_NN = ind_s_nbrs[findmax(weights_nbrs)[2]]
    val_x = value_array[ind_s_NN]

    return val_x
end

# used for GridInterpolations.jl indexing
function multi2single_ind(ind_m, sg)
    ind_s = 1
    for d in eachindex(ind_m)
        ind_s += (ind_m[d]-1) * prod(sg.state_grid.cut_counts[1:(d-1)])
    end

    return ind_s
end

# NOTE: functions with LazySets ---
# - to compute intersections without LazySets shapes, need numeric position/size of workspace, goal, obstacle, vehicle
#   - these are currently not passed into functions, since they're used to define LS shapes
#   - should be able to extract values from LS shapes, but requires interacting with them
#       - still avoids expensive set operations, so think it's fine
#   - otherwise would have to rewrite a fair amount of code to pass numeric values
#       - probably not that hard tbh, but try simpler method first

# checking with radial distances
# - for vehicle, need to find centerpoint for current state, use stored circumscription radius
#   - current functions take in x and veh, need to convert to centerpoint and raius within

# ---
# workspace checker
function in_workspace(x, env, veh)
    # veh_body = state_to_body(x, veh)
    veh_body = state_to_body_circle(x, veh)

    if issubset(veh_body, env.workspace)        # (!): LazySets
        return true
    end

    return false
end

# NEW
function in_workspace_SG(x, env, veh)
    # calculate centerpoint of vehicle body rectangle
    xp_c, yp_c = state_to_centerpoint_SG(x, veh)

    # extract dimensions of workspace rectangle
    x_coords = [vertex[1] for vertex in env.workspace.vertices]
    ws_x_max = maximum(x_coords)
    ws_x_min = minimum(x_coords)

    y_coords = [vertex[2] for vertex in env.workspace.vertices]
    ws_y_max = maximum(y_coords)
    ws_y_min = minimum(y_coords)

    # test if circle intersects workspace boundary
    if xp_c + veh.radius_vb > ws_x_max
        return false
    elseif xp_c - veh.radius_vb < ws_x_min
        return false
    elseif yp_c + veh.radius_vb > ws_y_max
        return false
    elseif yp_c - veh.radius_vb < ws_y_min
        return false
    else 
        return true
    end
end

# ---
# obstacle set checker
function in_obstacle_set(x, env, veh)
    # veh_body = state_to_body(x, veh)
    veh_body = state_to_body_circle(x, veh)

    for obstacle in env.obstacle_list
        # if isempty(intersection(veh_body, obstacle)) == false || isempty(intersection(obstacle, veh_body)) == false
        #     return true
        # end

        if isdisjoint(veh_body, obstacle) == false      # (!): LazySets
            return true
        end
    end

    return false
end

# NEW
function in_obstacle_set_SG(x, env, veh)
    # calculate centerpoint of vehicle body rectangle
    xp_c, yp_c = state_to_centerpoint_SG(x, veh)

    # check distance to each obstacle
    for obstacle_SG in env.obstacle_list_SG
        # get (centerpoint, radius) of obstacle
        xo_c = obstacle_SG[1]
        yo_c = obstacle_SG[2]
        radius_o = obstacle_SG[3]

        dist_vo = sqrt((xp_c - xo_c)^2 + (yp_c - yo_c)^2)
        if dist_vo < (radius_o + veh.radius_vb)
            return true
        end
    end

    return false
end

# ---
# target set checker
function in_target_set(x, env, veh)
    # state only inside goal region
    x_pos = Singleton(x[1:2])

    if issubset(x_pos, env.goal)        # (!): LazySets
        return true
    end

    # # full body inside goal region
    # veh_body = state_to_body(x, veh)

    # if issubset(veh_body, env.goal)
    #     return true
    # end

    return false
end

# NEW
function in_target_set_SG(x, env, veh)
    # calculate centerpoint of vehicle body rectangle
    xp_c, yp_c = state_to_centerpoint_SG(x, veh)
    println("[$xp_c, $yp_c]")
    println(veh.radius_vb)

    # check distance to goal
    xg_c = env.goal_SG[1]
    yg_c = env.goal_SG[2]
    radius_g = env.goal_SG[3]

    dist_vg = sqrt((xp_c - xg_c)^2 + (yp_c - yg_c)^2)
    println(dist_vg)

    if dist_vg < (radius_g) # (?): should vehicle radius be included here?
        println("[$xg_c, $yg_c]")
        println(radius_g)

        return true
    end

    return false
end

# ---

# vehicle body transformation function
function state_to_body(x, veh)
    # rotate body about origin by theta
    theta = x[3]
    rot_matrix = [cos(theta) -sin(theta); sin(theta) cos(theta)]
    body = linear_map(rot_matrix, veh.origin_body)

    # translate body from origin by [x, y]
    pos_vec = x[1:2]
    LazySets.translate!(body, pos_vec)

    return body
end

# vehicle body transformation function
function state_to_body_circle(x, veh)
    d = veh.origin_to_cent[1]

    xp_c = x[1] + d * cos(x[3])
    yp_c = x[2] + d * sin(x[3])

    body_circle = VPolyCircle([xp_c, yp_c], veh.radius_vb)

    return body_circle
end

# NEW
function state_to_centerpoint_SG(x, veh)
    d = veh.origin_to_cent[1]

    xp_c = x[1] + d * cos(x[3])
    yp_c = x[2] + d * sin(x[3])

    return xp_c, yp_c
end

# used to create circles as polygons in LazySets.jl
function VPolyCircle(cent_cir, r_cir)
    # number of points used to discretize edge of circle
    pts = 16

    # circle radius is used as midpoint radius for polygon faces (over-approximation)
    r_poly = r_cir/cos(pi/pts)

    theta_rng = range(0, 2*pi, length=pts+1)

    cir_vertices = [[cent_cir[1] + r_poly*cos(theta), cent_cir[2] + r_poly*sin(theta)] for theta in theta_rng]

    poly_cir = VPolygon(cir_vertices)

    return poly_cir
end