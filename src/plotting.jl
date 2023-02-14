# plotting.jl

# plot final value array over all velocity and heading states
function plot_HJB_value(value_array, env, veh, sg, heatmap_clim)
    # reshape value array into n-dimensional array
    value_array_m = reshape(value_array, sg.state_grid.cut_counts...)

    # plot HJB value function as a heat map
    for i3_plot in eachindex(sg.state_grid.cutPoints[3])
        for i4_plot in eachindex(sg.state_grid.cutPoints[4])
            plot_val = transpose(value_array_m[:, :, i3_plot, i4_plot])

            p_solver = heatmap(sg.state_grid.cutPoints[1], sg.state_grid.cutPoints[2], 
                plot_val, clim=(heatmap_clim, 1000),
                aspect_ratio=:equal, 
                xticks=0:4:20, yticks=0:4:20,
                size=(1000,800), dpi=300,
                xlabel="x-axis [m]", ylabel="y-axis [m]", 
                legend=:bottomright,
                legend_font_pointsize = 11,
                colorbar=true,
                colorbar_title = "Value",
                top_margin = -36*Plots.mm,
                bottom_margin = -8*Plots.mm,
                left_margin = 8*Plots.mm,
                right_margin = 6*Plots.mm)

            # workspace
            plot!(p_solver, env.workspace, 
                alpha=0.0, 
                linecolor=:black, linewidth=2, linealpha=1.0, 
                label="Workspace")

            # goal
            plot!(p_solver, env.goal, 
                color=:green, alpha=0.125, 
                linecolor=:green, linewidth=2, linealpha=1.0, 
                label="Goal")

            # obstacles
            if isempty(env.obstacle_list) == false
                plot!(p_solver, env.obstacle_list[1], 
                    color=:red, alpha=0.125, 
                    linecolor=:red, linewidth=2, linealpha=1.0, 
                    label="Obstacle")

                for obstacle in env.obstacle_list[2:end]
                    plot!(p_solver, obstacle, 
                        color=:red, alpha=0.125, 
                        linecolor=:red, linewidth=2, linealpha=1.0,
                        label="")
                end
            end

            # vehicle annotation
            xp = sg.state_grid.cutPoints[1][end] + 2.5
            yp = sg.state_grid.cutPoints[2][end]/2 + 1.0

            x_max = xp + sqrt((veh.origin_to_cent[1] + 1/2*veh.body_dims[1])^2 + (veh.origin_to_cent[2] + 1/2*veh.body_dims[2])^2)
            y_min = yp - sqrt((veh.origin_to_cent[1] + 1/2*veh.body_dims[1])^2 + (veh.origin_to_cent[2] + 1/2*veh.body_dims[2])^2)

            x = [xp, yp, sg.state_grid.cutPoints[3][i3_plot], sg.state_grid.cutPoints[4][i4_plot]]
            
            veh_body = state_to_body(x, veh)
                
            plot!(p_solver, [xp], [yp], 
                markercolor=:blue, markershape=:circle, markersize=2, markerstrokewidth=0, 
                label="")

            plot!(p_solver, veh_body, 
                color=:blue, alpha=0.125, 
                linecolor=:blue, linewidth=2, linealpha=1.0, 
                label="Vehicle")

            plot!(p_solver, [x_max], [yp], markercolor=:white, label="")
            plot!(p_solver, [xp], [y_min], markercolor=:white, label="")

            theta_deg = round(rad2deg(x[3]), digits=1)
            annotate!(xp, yp-2.5, text("th [deg]:\n$theta_deg", 14))

            v = round(x[4], digits=2)
            annotate!(xp, yp-4.5, text("v [m/s]:\n$v", 14))

            display("image/png", p_solver)
        end
    end
end

# plot current value array at a given velocity/heading at each step in the solving process
function plot_HJB_growth(value_array, heatmap_clim, step, env, veh)
    # reshape value array into n-dimensional array
    value_array_m = reshape(value_array, sg.state_grid.cut_counts...)

    i3_plot = 19
    i4_plot = 1
    plot_val = transpose(value_array_m[:, :, i3_plot, i4_plot])
    
    # plot HJB value function as a heat map
    p_growth = heatmap(sg.state_grid.cutPoints[1], sg.state_grid.cutPoints[2], 
                plot_val, clim=(0, heatmap_clim),
                aspect_ratio=:equal, 
                xticks=0:4:20, yticks=0:4:20,
                size=(800,800), dpi=300,
                xlabel="x-axis [m]", ylabel="y-axis [m]", 
                legend=:bottomright,
                legend_font_pointsize = 11,
                colorbar=true,
                colorbar_title = "Value",
                top_margin = -36*Plots.mm,
                bottom_margin = -8*Plots.mm,
                left_margin = 8*Plots.mm,
                right_margin = 6*Plots.mm)

    # workspace
    plot!(p_growth, env.workspace, 
    alpha=0.0, 
    linecolor=:black, linewidth=2, linealpha=1.0, 
    label="Workspace")

    # goal
    plot!(p_growth, env.goal, 
        color=:green, alpha=0.125, 
        linecolor=:green, linewidth=2, linealpha=1.0, 
        label="Goal")

    # obstacles
    if isempty(env.obstacle_list) == false
        plot!(p_growth, env.obstacle_list[1], 
            color=:red, alpha=0.125, 
            linecolor=:red, linewidth=2, linealpha=1.0, 
            label="Obstacle")

        for obstacle in env.obstacle_list[2:end]
            plot!(p_growth, obstacle, 
                color=:red, alpha=0.125, 
                linecolor=:red, linewidth=2, linealpha=1.0,
                label="")
        end
    end

    # vehicle annotation
    xp = sg.state_grid.cutPoints[1][end] + 4.5
    yp = sg.state_grid.cutPoints[2][end]/2  - 0.5

    x_max = xp + sqrt((veh.origin_to_cent[1] + 1/2*veh.body_dims[1])^2 + (veh.origin_to_cent[2] + 1/2*veh.body_dims[2])^2)
    y_min = yp - sqrt((veh.origin_to_cent[1] + 1/2*veh.body_dims[1])^2 + (veh.origin_to_cent[2] + 1/2*veh.body_dims[2])^2)

    x = [xp, yp, sg.state_grid.cutPoints[3][i3_plot], sg.state_grid.cutPoints[4][i4_plot]]
    
    veh_body = state_to_body(x, veh)
        
    plot!(p_growth, [xp], [yp], 
        markercolor=:blue, markershape=:circle, markersize=2, markerstrokewidth=0, 
        label="")

    plot!(p_growth, veh_body, 
        color=:blue, alpha=0.125, 
        linecolor=:blue, linewidth=2, linealpha=1.0, 
        label="Vehicle")

    plot!(p_growth, [x_max], [yp], markercolor=:white, label="")
    plot!(p_growth, [xp], [y_min], markercolor=:white, label="")

    theta_deg = round(rad2deg(x[3]), digits=1)
    annotate!(xp, yp+1.5, text("th [deg]:\n$theta_deg", 14))

    v = round(x[4], digits=2)
    annotate!(xp, yp-1.5, text("v [m/s]:\n$v", 14))

    # step count
    annotate!(xp, yp-4.5, text("step:\n$(step-1)", 14))

    display(p_growth)
end

# plot paths from planner
function plot_HJB_path(x_path_list, x_subpath_list, env, veh, linez_clim, label_list)
    p_planner = plot(aspect_ratio=:equal,
        size=(800,800), dpi=300,
        xticks=0:4:20, yticks=0:4:20, 
        xlabel="x-axis [m]", ylabel="y-axis [m]",
        legend=:bottomright,
        top_margin = -36*Plots.mm,
        bottom_margin = -8*Plots.mm,
        left_margin = 8*Plots.mm,
        right_margin = 6*Plots.mm)

    # workspace
    plot!(p_planner, env.workspace, 
        alpha=0.0, 
        linecolor=:black, linewidth=2, linealpha=1.0, 
        label="")
    
    # goal
    plot!(p_planner, env.goal, 
        color=:green, alpha=0.125, 
        linecolor=:green, linewidth=2, linealpha=1.0, 
        label="Goal")

    # obstacles
    if isempty(env.obstacle_list) == false
        plot!(p_planner, env.obstacle_list[1], 
            color=:red, alpha=0.125, 
            linecolor=:red, linewidth=2, linealpha=1.0, 
            label="Obstacle")

        for obstacle in env.obstacle_list[2:end]
            plot!(p_planner, obstacle, 
                color=:red, alpha=0.125, 
                linecolor=:red, linewidth=2, linealpha=1.0,
                label="")
        end
    end

    for ip in axes(x_path_list, 1)
        x_path = x_path_list[ip]
        x_subpath = x_subpath_list[ip]

        # shift velocity up one step to make line_z look right
        linez_velocity = zeros(length(x_subpath))
        for kk in 1:(length(x_subpath)-1)
            linez_velocity[kk] = x_subpath[kk+1][4]
        end

        # subpath lines
        plot!(p_planner, getindex.(x_subpath,1), getindex.(x_subpath,2),
            linez=linez_velocity, clim=(0, linez_clim), colorbar_title="Velocity [m/s]",
            linewidth = 2,
            label="")

        # path points
        plot!(p_planner, getindex.(x_path,1), getindex.(x_path,2),
            linewidth = 0, linealpha=0.0,
            markershape=:circle, markersize=3.0, markerstrokewidth=0, 
            label=label_list[ip])

        # start position
        plot!(p_planner, [x_path[1][1]], [x_path[1][2]], 
            markershape=:circle, markersize=3, markerstrokewidth=0, 
            label="")

        veh_body = state_to_body(x_path[1], veh)
        plot!(p_planner, veh_body, alpha=0.0,
            linecolor=:black, linewidth=2, linealpha=1.0, label="")

        # end position
        plot!(p_planner, [x_path[end][1]], [x_path[end][2]], 
            markershape=:circle, markersize=3, markerstrokewidth=0, 
            label="")

        veh_body = state_to_body(x_path[end], veh)
        plot!(p_planner, veh_body, alpha=0.0, 
            linecolor=:black, linewidth=2, linealpha=1.0, label="")
    end

    display("image/png", p_planner)
end

function plot_path_value(val_path_list, Dt)
    p_value = plot(
        xlabel="Time [sec]", ylabel="Value",
        legend=:topleft)

    label_list = ["Optimal", "Optimal Reactive", "Approx Reactive"]
    for ip in axes(val_path_list, 1)
        val_path = val_path_list[ip]
        tspan = 0:Dt:(length(val_path)-1)*Dt
        
        plot!(p_value, tspan, val_path,
            linewidth = 2.0,
            markershape=:circle, markersize=2.5, markerstrokewidth=0, 
            label=label_list[ip])
    end

    display(p_value)
end