# utils.jl

# 4th-order Runge-Kutta integration scheme
function runge_kutta_4(x_k, u_k, dt::Float64, EoM, veh, sg)    
    w1 = EoM(x_k, u_k, veh)
    w2 = EoM(x_k + w1*dt/2, u_k, veh)
    w3 = EoM(x_k + w2*dt/2, u_k, veh)
    w4 = EoM(x_k + w3*dt, u_k, veh)

    x_k1 = x_k + (1/6)*dt*(w1 + 2*w2 + 2*w3 + w4)

    x_k1 = mod_state_angle(x_k1, sg)

    return x_k1
end

# adjust angles within [-pi,pi] bounds
function mod_state_angle(x, sg)
    for d in eachindex(x)
        if sg.angle_wrap_array[d] == true
            x[d] = x[d] % (2*pi)
            x[d] > pi ? x[d] -= 2*pi : x[d] -= 0
        end
    end

    return x
end

function interp_value(x, value_array, sg)
    # check if current state is within state space
    for d in eachindex(x)
        if x[d] < sg.state_grid.cutPoints[d][1] || x[d] > sg.state_grid.cutPoints[d][end]
            val_itp = 1e5
            return val_itp
        end
    end

    # interpolate value at given state
    val_itp = interpolate(sg.state_grid, value_array, x)

    return val_itp
end

# used for GridInterpolations.jl indexing
function multi2single_ind(ind_m, sg)
    ind_s = 1
    for d in eachindex(ind_m)
        ind_s += (ind_m[d]-1)*prod(sg.state_grid.cut_counts[1:(d-1)])
    end

    return ind_s
end

# workspace checker
function in_workspace(x, env, veh)
    veh_body = state_to_body(x, veh)

    if issubset(veh_body, env.workspace)
        return true
    end
        
    return false
end

# obstacle set checker
function in_obstacle_set(x, env, veh)
    veh_body = state_to_body(x, veh)
    
    for obstacle in env.obstacle_list
        if isempty(intersection(veh_body, obstacle)) == false
            return true
        end
    end

    return false
end

# target set checker
function in_target_set(x, env::Environment, veh::Vehicle)
    veh_body = state_to_body(x, veh)

    if issubset(veh_body, env.goal)
        return true
    end

    return false
end

# vehicle body transformation function
function state_to_body(x, veh)
    # rotate body about origin by theta
    rot_matrix = [cos(x[3]) -sin(x[3]); sin(x[3]) cos(x[3])]
    body = linear_map(rot_matrix, veh.origin_body)

    # translate body from origin by [x, y]
    trans_vec = x[1:2]
    LazySets.translate!(body, trans_vec)

    return body
end

# used to create circles as polygons in LazySets.jl
function circle2vpolygon(cent_cir, r_cir)
    # number of points used to discretize edge of circle
    pts = 12

    # circle radius is used as midpoint radius for polygon faces (over-approximation)
    r_poly = r_cir/cos(pi/pts)

    theta_rng = range(0, 2*pi, length=pts+1)

    cir_vertices = [[cent_cir[1] + r_poly*cos(theta), cent_cir[2] + r_poly*sin(theta)] for theta in theta_rng]
    
    poly_cir = VPolygon(cir_vertices)
    
    return poly_cir
end