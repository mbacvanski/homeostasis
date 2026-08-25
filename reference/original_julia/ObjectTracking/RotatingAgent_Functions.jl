using Agents, LinearAlgebra, InteractiveDynamics, GLMakie, Statistics, Meshes, Distributions, DataFrames, StatsBase

mutable struct Agent <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    type::Symbol # :agent, :stim
    complexPos::ComplexF64
    speed::Float64
    heading::Float64
    sens_angles::Vector{Float64}
    sens_Cpos::NTuple{2, ComplexF64}
    degree::Int64
end

global learn_on = 1
global acts_neg = 0
global agent_radius = .5
global stim_radius = 1

global save_data = true

global ts_heading = []
global ts_spikes = []
global ts_acts = []
global ts_sens = []
global ts_eff = []
global ts_stim = []

_Agent(id, pos, heading, sens_angles) = Agent(id, pos, :agent, pos[1] + pos[2]im, 0, heading, sens_angles, ( agent_radius * ( cos(heading+deg2rad(30)) + sin(heading+deg2rad(30))im), agent_radius * ( cos(heading-deg2rad(30)) + sin(heading-deg2rad(30))im)), 0)
Stimulus(id, pos) = Agent(id, pos, :stim, pos[1] + pos[2]im, 1, 0, [], (0+0im,0+0im), 0)

### functions

# function get_plot_vals(degree, heading)
#     x = np.cos(np.radians(degree))
#     y = np.sin(np.radians(degree))
    
#     sLdeg = heading + sens_offset
#     if sLdeg > 360
#         sLdeg = sLdeg - 360
#     end

#     sRdeg = heading - sens_offset
#     if sRdeg < 0
#         sRdeg = 360 + sRdeg
#     end
        
#     sLx = .5*np.cos(np.radians(sLdeg)) 
#     sLy = .5*np.sin(np.radians(sLdeg)) 
#     sLpos = np.array([sLx,sLy])
    
#     sRx = .5*np.cos(np.radians(sRdeg)) 
#     sRy = .5*np.sin(np.radians(sRdeg))
#     sRpos = np.array([sRx, sRy])
    
#     return x, y, sLx, sLy, sRx, sRy
# end

function gaussian(x)
    exp(-((x^2)/10))
end

function move_stim(stim, direction,model)
    if direction == 1
        stim.degree += 1 #stim_speed
    else
        stim.degree -= 1 #stim_speed
    end
        
    if stim.degree > 360
        stim.degree -= 360
    elseif stim.degree < 0
        stim.degree += 360
    end

    stim.complexPos = stim_radius * (cos(deg2rad(stim.degree)) + sin(deg2rad(stim.degree))im)

    move_agent!(stim, (real(stim.complexPos)+1.5, imag(stim.complexPos)+1.5),model)
end

function rotate_agent(output_acts, agent)
    diff = (output_acts[1] - output_acts[2])*movement_amp
    _heading = rad2deg(agent.heading)
    _heading += diff
    if _heading > 360
        _heading -= 360
    elseif _heading < 0
        _heading += 360
    end
    agent.heading = deg2rad(_heading)
end

function get_input_acts(agent,stim)
    
    agent_stim_angle = rad2deg(rem2pi(angle(stim.complexPos) - agent.heading, RoundNearest))
    
    sens_dists = abs.(agent.sens_angles .- agent_stim_angle)
    sens_acts = zeros(length(sens_degrees))
    sens_acts = gaussian.(sens_dists)
    sens_acts[sens_dists .<= 4] .= 1 #- sens_dists/60
        
    sens_acts
end

function get_acts(acts,leak,spikes,wmat,input,input_wmat,targets)
    
    acts = (acts .* (1-leak) .+ sum(input .* input_wmat, dims = 1)' .+ sum(spikes[] .* wmat[], dims = 1)')[:,1]

    if acts_neg == 0
        acts[acts .< 0] .= 0
    end
    
    prev_spikes = copy(spikes[])
    thresholds = targets .* 2
    spikes[][acts .>= thresholds] .= 1
    spikes[][acts .< thresholds] .= 0
    
    acts[spikes[] .== 1] .-= thresholds[spikes[] .== 1]
       
    errors = acts .- targets
    
    [acts, spikes[], errors, prev_spikes]
end

function learning(learn_on,link_mat,spikes,prev_spikes, errors,wmat,targets)
    
    active_neighbors=abs.(link_mat)
    active_neighbors[prev_spikes .== 0,:] .= 0
    d_wmat = copy(active_neighbors)
    active_neighbors = sum(active_neighbors,dims=1)
    
    if learn_on==1
        if sum(active_neighbors) >0
            d_wmat = (errors' .* d_wmat)
            d_wmat=(d_wmat ./ active_neighbors)
            d_wmat[isinf.(d_wmat)] .= 0 
            d_wmat[isnan.(d_wmat)] .= 0 
            wmat[] .-= d_wmat
        end
            
        targets .+= (errors .* lrate_targ) 
        targets[targets .< targ_min] .= targ_min
    end
            
    [wmat[], targets]
end

function model_step!(model)
    global acts
    global spikes
    global targets
    global errors
    global prev_spikes
    global wmat
    global input_wmat
    global direction

    agent = model[1]
    stim = model[2]

    if mod(model.n, 360*2) == 0 && model.n != 0
        if direction == 1
            direction = 2
        else
            direction = 1
        end
    end

    input = get_input_acts(agent, stim)
    model.inputs = input

    acts, spikes[], errors, prev_spikes = get_acts(acts,leak,spikes,wmat,input,input_wmat,targets)

    output_acts = (sum(spikes[] .* output_wmat,dims = 1) ./ sum(output_wmat, dims=1))[1,:]
    model.outputs = output_acts

    wmat[], targets = learning(learn_on,link_mat,spikes,prev_spikes, errors,wmat,targets)
    # model.weights = wmat

    #saving data
    if save_data == true
        push!(ts_spikes, Tuple(Int64(x) for x in spikes[]))
        push!(ts_acts, Tuple(Float64(x) for x in acts))
        push!(ts_heading, agent.heading)

        push!(ts_sens, Tuple(Float64(x) for x in input))
        push!(ts_eff, Tuple(Float64(x) for x in output_acts))
        push!(ts_stim, deg2rad(stim.degree))
    end

    move_stim(stim, direction,model)
    rotate_agent(output_acts, agent)

    model.mean_act = mean(acts)
    model.mean_err = mean(errors)
    model.mean_targ = mean(targets)

    model.n += 1
end