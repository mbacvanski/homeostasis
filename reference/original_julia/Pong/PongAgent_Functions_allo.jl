#using Agents, LinearAlgebra, InteractiveDynamics, GLMakie, Statistics, Meshes, Distributions, DataFrames, StatsBase, Random
using Agents, LinearAlgebra, InteractiveDynamics, Observables, Statistics, Meshes, Distributions, DataFrames, StatsBase, Random

import GeometryBasics as gb

mutable struct Agent <: AbstractAgent
    id::Int
    pos::NTuple{2,Float64}
    type::Symbol # :agent, :stim
    complexPos::ComplexF64
    speed::Float64
    heading::Float64
    sens_degrees::Vector{Float64}
    sens_Cpos::NTuple{2, ComplexF64}
    degree::Int64
    hits::Vector{Int64}
end

global learn_on = 1
global acts_neg = 1
global agent_radius = 10
global ball_radius = 15

_Agent(id, pos, heading, sens_degrees, hits) = Agent(id, pos, :agent, pos[1] + pos[2]im, 0, heading, sens_degrees, ( agent_radius * ( cos(heading+deg2rad(30)) + sin(heading+deg2rad(30))im), agent_radius * ( cos(heading-deg2rad(30)) + sin(heading-deg2rad(30))im)), 0, hits)
Stimulus(id, pos) = Agent(id, pos, :stim, pos[1] + pos[2]im, 1, 0, [], (0+0im,0+0im), 0, [0])

function gaussian(x)
    exp(-((x^2)/.1))
end

# global dx = rand(Uniform(-10,10))#
# global dy = rand(Uniform(-10,10))#StatsBase.sample([-stim_speed,stim_speed])

global xSpeed = 5 #rand(Uniform(3,10))
global ySpeed = 5 #rand(Uniform(3,10))
global dx = -xSpeed#StatsBase.sample([-xSpeed,xSpeed])
global dy = StatsBase.sample([-ySpeed,ySpeed])
function move_stim(stim, agent, dx, dy,model)

    newPos = [stim.pos[1] + dx, stim.pos[2] + dy]
    
    s1 = Segment(Meshes.Point2(stim.pos),Meshes.Point2(newPos))
    s2 = Segment(Meshes.Point2(agent.pos[1],agent.pos[2]-paddle_radius),Meshes.Point2(agent.pos[1],agent.pos[2]+paddle_radius))

    res = s1 ∩ s2
    if !isnothing(res)

        #distₓ = 
        dist₀ = abs((coordinates(res)[1]+ coordinates(res)[2]im) - (stim.pos[1]+stim.pos[2]im)) #Agents.euclidean_distance(res, stim.pos,model)
        dist₁ = abs((newPos[1]+ newPos[2]im) - (stim.pos[1]+stim.pos[2]im))
        diff = dist₁ - dist₀
        newPos[1] += diff + 5
        dx *= -1
        push!(model[1].hits, 1)
    end
    

    if newPos[1] >= 995
        newPos[1] = 995
        # newPos[1] = 500
        # newPos[2] = rand(Uniform(1,499))
        dx *= -1 #StatsBase.sample([-5,5])
    elseif newPos[1] <= 0
        newPos[1] = 995
        newPos[2] = rand(Uniform(1,499))


        global xSpeed = 5#rand(Uniform(4,8))
        global ySpeed = 5#rand(Uniform(4,8))
        global dx = -xSpeed
        global dy = StatsBase.sample([-ySpeed,ySpeed])  

        push!(model[1].hits, 0)
    elseif newPos[2] >= 495
        newPos[2] = 495
        dy *= -1
    elseif newPos[2] <= 5
        newPos[2] = 5
        dy *= -1
    end

    move_agent!(stim, (newPos[1],newPos[2]),model)
    stim.complexPos = stim.pos[1] + stim.pos[2]im
    [dx,dy]
    
end

function move_agent(output_acts, agent)
    diffL = (output_acts[1] - output_acts[2])*movement_amp
    # diffR = (output_acts[3]-output_acts[4])*movement_amp
    if agent.id == 1
        newY = agent.pos[2]+diffL
        if newY >= 500 - paddle_radius
            newY = 500 - paddle_radius
        elseif newY <= 0 +paddle_radius
            newY = 0+paddle_radius
        end
        move_agent!(agent, (agent.pos[1],newY),model)
        agent.complexPos = agent.pos[1] + agent.pos[2]im
    # elseif agent.id == 2
    #     newY = agent.pos[2]+diffR
    #     if newY >= 500 -40
    #         newY = 500 -40
    #     elseif newY <= 0 +40
    #         newY = 0+40
    #     end
    #     move_agent!(agent, (agent.pos[1],newY),model)
    end
end

function get_input_acts(stim,agent)

    stimY = imag(stim.complexPos)
    angDists = abs.(sens_degrees .- stimY)
    stimDist = 1-abs(stim.complexPos-agent.complexPos)/900
    # if stimDist < .5
    #     stimDist = .5
    # end

    sens_acts = float(angDists .<= 5) #.* stimDist


    ####
    #sens_acts = gaussian.(angDists)

    # stimDist_L = stim.pos[2] - model[1].pos[2]

    # sensDists_L = abs.(sens_degrees .- stimDist_L)
    # sensDists = reduce(vcat, sensDists_L)
    
    # sens_acts = gaussian.(sensDists)#*(1 - stim.pos[1]/900)

    # if stim.pos[1] < model[1].pos[1]
    #     sens_acts = zeros(length(sens_degrees))
    # end
        
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
    global dx
    global dy

    agent=model[1]
    stim = model[2]
    input = get_input_acts(stim,agent)
    model.inputs = input
    acts, spikes[], errors, prev_spikes = get_acts(acts,leak,spikes,wmat,input,input_wmat,targets)
    output_acts = (sum(spikes[] .* output_wmat,dims = 1) ./ sum(output_wmat, dims=1))[1,:]
    model.outputs = output_acts
    wmat[], targets = learning(learn_on,link_mat,spikes,prev_spikes, errors,wmat,targets)
    # model.weights = wmat

    dx, dy = move_stim(stim, agent,dx,dy,model)


    for agent in allagents(model) ## only need this loop for two pong paddles, irrelevant now
        move_agent(output_acts, agent)

    end

    model.mean_act = mean(acts)
    model.mean_err = mean(errors)
    model.mean_targ = mean(targets)

    model.n += 1
end