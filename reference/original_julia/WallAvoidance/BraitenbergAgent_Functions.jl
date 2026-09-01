using Agents, LinearAlgebra, InteractiveDynamics, GLMakie, Statistics, Meshes, Distributions, DataFrames, StatsBase
import GeometryBasics as gb

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
    tStep::Int64
end

global learn_on = 1
global acts_neg = 0
global agent_radius = .5
global stim_radius = 1

global ts_pos = []
global ts_hits = []
global ts_spikes = []
global ts_acts = []
global ts_heading = []
global ts_sens = []
global ts_eff = []

_Agent(id, pos, heading, sens_angles) = Agent(id, pos, :agent, pos[1] + pos[2]im, 0, heading, sens_angles, ( agent_radius * ( cos(heading+deg2rad(30)) + sin(heading+deg2rad(30))im), agent_radius * ( cos(heading-deg2rad(30)) + sin(heading-deg2rad(30))im)), 0, 0)
Stimulus(id, pos) = Agent(id, pos, :stim, pos[1] + pos[2]im, 1, 0, [], (0+0im,0+0im), 0)

### functions


function move_agent(output_acts, agent,model)

    # translation
    vel_centre = (output_acts[1]+output_acts[2])/2
    pos_centre = vel_centre .* [cos(agent.heading), sin(agent.heading)]
    # agent.pos[1] += pos_centre[1]
    # agent.pos[2] += pos_centre[2]
    
    # rotation
    omega = (output_acts[2]-output_acts[1])/(2*agent_radius)
    agent.heading += omega

    randomRot = 0
    newPos = [agent.pos[1] + pos_centre[1], agent.pos[2] + pos_centre[2]]
    if newPos[1] > (15 - agent_radius)
        newPos[1] = 15 - agent_radius
        randomRot =1
    end
    if newPos[1] < agent_radius
        newPos[1] = agent_radius
        randomRot =1
    end
    if newPos[2] > 15 - agent_radius
        newPos[2] = 15 - agent_radius
        randomRot =1
    end
    if newPos[2] < agent_radius
        newPos[2] = agent_radius
        randomRot =1
    end
    if randomRot == 1
        agent.heading += deg2rad(StatsBase.sample([-45,45])) #rand(Uniform(-deg2rad(45),deg2rad(45)))
        agent.heading = rem2pi(agent.heading, RoundNearest)
        push!(ts_hits, 1)
    else
        push!(ts_hits, 0)
    end

    move_agent!(agent, (newPos[1],newPos[2]),model)
    # agent.pos = newPos

end

box_faces = [
    Segment((0,0),(15, 0)),
    Segment((15,0),(15, 15)),
    Segment((15,15),(0, 15)),
    Segment((0,15),(0, 0)),
]
maxDist = sqrt(15^2 + 15^2)

function get_input_acts(agent)

    pos = agent.pos
    heading = agent.heading
    sL_ang = agent.heading + deg2rad(45)
    sR_ang = agent.heading - deg2rad(45)

    sL_pos = (agent_radius * cos(heading+deg2rad(45)) + pos[1], agent_radius * sin(heading+deg2rad(45)) + pos[2])
    sR_pos = (agent_radius * cos(heading-deg2rad(45)) + pos[1], agent_radius * sin(heading-deg2rad(45)) + pos[2])

    sL_endX = 100*(cos(sL_ang))
    sL_endY = 100*(sin(sL_ang))

    sL_end = sL_pos .+ (sL_endX, sL_endY)

    sR_endX = 100*(cos(sR_ang))
    sR_endY = 100*(sin(sR_ang))

    sR_end = sR_pos .+ (sR_endX, sR_endY)

    sL_seg = Segment(sL_pos,sL_end)
    sR_seg = Segment(sR_pos,sR_end)

    leftIntersect = nothing
    for face in box_faces
        leftIntersect = sL_seg ∩ face
        if !isnothing(leftIntersect)
            leftIntersect = coordinates(leftIntersect)
            break
        end
    end
    sL_dist = sL_pos .- leftIntersect
    sL_dist = sqrt(sL_dist[1]^2 + sL_dist[2]^2)

    rightIntersect = nothing
    for face in box_faces
        rightIntersect = sR_seg ∩ face
        if !isnothing(rightIntersect)
            rightIntersect = coordinates(rightIntersect)
            break
        end
    end
    sR_dist = sR_pos .- rightIntersect
    sR_dist = sqrt(sR_dist[1]^2 + sR_dist[2]^2)

    if noise > 0
        sL_act = 1 - sL_dist/maxDist + rand(Uniform(-noise,noise))
        sR_act = 1 - sR_dist/maxDist + rand(Uniform(-noise,noise))
    else
        sL_act = 1 - sL_dist/maxDist
        sR_act = 1 - sR_dist/maxDist
    end

    if perturb == 1 && (model[1].tStep > 1000)
        sens_acts = [sR_act*2, sL_act*2]
    else
        sens_acts = [sL_act, sR_act]
    end
        
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

    for agent in allagents(model)
        
        agent.tStep += 1
        input = get_input_acts(agent)
        model.inputs = input

        acts, spikes[], errors, prev_spikes = get_acts(acts,leak,spikes,wmat,input,input_wmat,targets)

        output_acts = (sum(spikes[] .* output_wmat,dims = 1) ./ sum(output_wmat, dims=1))[1,:]
        # output_acts[1] += rand(Uniform(0,.25))
        # output_acts[2] += rand(Uniform(0,.25))
        model.outputs = output_acts

        #saving data
        push!(ts_spikes, Tuple(Int64(x) for x in spikes[]))
        push!(ts_acts, Tuple(Float64(x) for x in acts))
        push!(ts_pos, agent.pos)

        push!(ts_heading, (agent.heading))
        push!(ts_sens, Tuple(Float64(x) for x in input))
        push!(ts_eff, Tuple(Float64(x) for x in output_acts))

        wmat[], targets = learning(learn_on,link_mat,spikes,prev_spikes, errors,wmat,targets)
        # model.weights = wmat

        move_agent(output_acts, agent,model)

    end


    model.mean_act = mean(acts)
    model.mean_err = mean(errors)
    model.mean_targ = mean(targets)

    model.n += 1
end