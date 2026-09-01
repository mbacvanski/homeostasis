using SimpleWeightedGraphs, Graphs
import GraphPlot
include("./WallAvoidance/BraitenbergAgent_Functions.jl");

## params
plotIter = 3600*2
nnodes = 200
p_link = .1
leak = .25 #
leaktype = 1 #1 = leak a percentage, or 2 = leak a constant rate
lrate_wmat = .01
lrate_targ = .01
targ_min = 1.0 
movement_amp = 10
input_amp = 4

global learn_on = 1

global acts_neg = 1

global perturb = 1

global noise = 0

####
sens_degrees = [45, -45]#vcat(sort(collect(range(start=-61,stop=63,step=4)).+30,rev=true), sort(collect(range(start=-60,stop=64,step=4)).-30,rev=true))
sensColors= [:red, :blue]#reduce(vcat,[repeat([:red],32), repeat([:blue], 32)])

sensory_nodes = StatsBase.sample(range(1,nnodes),Weights(repeat([1],nnodes)),Int(round(nnodes*p_link)))
global input_wmat=zeros((length(sens_degrees),nnodes))
for row in range(1,size(input_wmat)[1])
    #for col in sensory_nodes
    for col in range(1,nnodes)
        input_wmat[row,col]=StatsBase.sample([0,input_amp],Weights([1-p_link,p_link]))
    end
end

global link_mat = zeros((nnodes,nnodes))
inhibitory_nodes=[]
for row in range(1,size(link_mat)[1])
    inhibitory = StatsBase.sample([0,1],Weights([.75,.25]))
    inhibitory = StatsBase.sample([0,1],Weights([1,0]))
    if inhibitory == 1
        push!(inhibitory_nodes,row)
    end
    for col in range(1,size(link_mat)[2])
        if row == col
            continue
        elseif inhibitory == 1
            link_mat[row,col] = StatsBase.sample([0,-1],Weights([1-p_link,p_link]))
        else
            link_mat[row,col] = StatsBase.sample([0,1],Weights([1-p_link,p_link]))
        end
    end
end
        
global wmat= Observable(zeros((nnodes,nnodes)))
for row in range(1,size(wmat[])[1])
    for col in range(1,size(wmat[])[2])
        if link_mat[row,col] == 1
            wmat[][row,col] =  rand(Normal(input_amp,.1))
        elseif link_mat[row,col] == -1
            wmat[][row,col] =  rand(Normal(-input_amp,.1))
        end
    end
end
startWmat = copy(wmat[])

effector_nodes=[]
global output_wmat=zeros((nnodes,2))
for row in range(1,size(output_wmat)[1])
    for col in range(1,size(output_wmat)[2])
        output_wmat[row,col]=StatsBase.sample([0,1],Weights([1-p_link,p_link]))
        if output_wmat[row,col] != 0
            push!(effector_nodes,row)
        end
    end
end


######

global spikes = Observable(zeros(nnodes))
global targets = repeat([targ_min],nnodes)
global acts = zeros(nnodes)
global inputs = zeros(length(sens_degrees))
global output_acts = zeros(2)

global n = 0
global direction = 1
global mean_act = 0.0
global mean_err = 0.0
global mean_targ = mean(targets)

model = ABM(Agent, 
ContinuousSpace((15,15), 
periodic = false); 
properties = Dict(
    :n => n,
    :direction => direction,
    :mean_act => mean_act,
    :mean_err => mean_err,
    :mean_targ => mean_targ,
    :inputs => inputs,
    :outputs => output_acts,
    :weights => wmat,
),
scheduler = Schedulers.fastest
)

add_agent_pos!(
    _Agent(
        nextid(model),
        (7.5,7.5),
        π/2,
        sens_degrees,

    ),
    model
)

# add_agent_pos!(
#     Stimulus(
#         nextid(model),
#         ( stim_radius * cos(0)+1.5, stim_radius * sin(0)+1.5),

#     ),
#     model
# )

#### plotting
adata = [:pos, :heading, :degree]
mdata = [:mean_act, :mean_err, :mean_targ, :inputs, :outputs]

params = Dict(   
)

fig, ax, abmobs = abmplot(model; 
    dummystep, model_step!,
    add_controls = true, enable_inspection = true,
    adata, mdata, figure = (; resolution = (2400,1600))
)

sensors_layout = fig[1,end+1] = GridLayout()
acts_layout = fig[1,end+1] = GridLayout()

text_layout = fig[2, end-1]
ax_text = Axis(text_layout[1,1])
xlims!(ax_text, (-1,1))
ylims!(ax_text, (-1,1))

network_layout = fig[3, end-2] = GridLayout()
effectors_layout = fig[3,end-1] = GridLayout()
weights_layout = fig[3,end] = GridLayout()

ax_sensors = Axis(sensors_layout[1,1];
    backgroundcolor = :lightgrey, ylabel = "Value", xticks = (1:length(sens_degrees)))
xlims!(ax_sensors, (0,length(sens_degrees)+1))
ylims!(ax_sensors, (0,1.2))

ax_acts = Axis(acts_layout[1,1]; backgroundcolor = :lightgrey, ylabel = "Value", xticks = (1:3, ["meanActs","meanErrs", "meanTargs"]), xticklabelrotation = pi/8)
xlims!(ax_acts, (0,4))
ylims!(ax_acts, (-3,3))

ax_network = Axis(network_layout[1,1])

ax_effectors = Axis(effectors_layout[1,1];
    backgroundcolor = :lightgrey, ylabel = "Value", xticks = (1:2, ["Left Turn", "Right Turn"]))
xlims!(ax_effectors, (0,3))
ylims!(ax_effectors, (0,1))

ax_weights = Axis(weights_layout[1,1]; backgroundcolor = :lightgrey)

####
sL = @lift begin
    a = $(abmobs.adf)
    b = last(filter(:id => n -> n == 1, a))
    heading = getindex(b.heading)[1]
    x = getindex(b.pos[1])
    y = getindex(b.pos[2])
    sL = (agent_radius * cos(heading+deg2rad(45)) + x, agent_radius * sin(heading+deg2rad(45)) + y)#( agent_radius * ( cos(heading+deg2rad(30)) + sin(heading+deg2rad(30))im), agent_radius * ( cos(heading-deg2rad(30)) + sin(heading-deg2rad(30))im))
    [sL]
end

sR = @lift begin
    a = $(abmobs.adf)
    b = last(filter(:id => n -> n == 1, a))
    heading = getindex(b.heading)[1]
    x = getindex(b.pos[1])
    y = getindex(b.pos[2])
    sR = (agent_radius * cos(heading-deg2rad(45)) + x, agent_radius * sin(heading-deg2rad(45)) + y)
    [sR]
end

agentLoc = @lift begin
    a = $(abmobs.adf)
    b = last(filter(:id => n -> n == 1, a))
    pos = b.pos
    # x = getindex(b.pos[1])
    # y = getindex(b.pos[2])
    # z = (x,y)
    # z
    Circle(GLMakie.Point2f(pos), agent_radius)
end

positions = @lift begin
    a = $(abmobs.adf)
    b = unique(filter(:id => n -> n == 1, a))
    b.pos
end


meanAct = @lift begin
    a = $(abmobs.mdf)
    b = last(a)
    [b.mean_act]
end

meanErr = @lift begin
    a = $(abmobs.mdf)
    b = last(a)
    [b.mean_err]
end

meanTarg = @lift begin
    a = $(abmobs.mdf)
    b = last(a)
    [b.mean_targ]
end

sensActs = @lift begin
    a = $(abmobs.mdf)
    b = last(a)
    b.inputs
end

step_ = @lift begin
    a = $(abmobs.adf)
    b = last(a)
    string(b.step[1])
end

text!(ax_text, step_, position=(0,0),align = (:center, :center),fontsize=100)

effectorActs = @lift begin
    a = $(abmobs.mdf)
    b = last(a)
    b.outputs
end

colors = @lift begin
    [x == 0 ? :blue : :yellow for x in $spikes]
end

_weights2 = @lift begin
    x =reduce(vcat, $(wmat))
    x[x .!= 0]
end

####
scatterlines!(ax, positions; color = :black, markersize = 5, markeralpha = .01)

poly!(ax, agentLoc, color = :pink)



scatter!(ax, sL ; markersize = 10, color = :red)
scatter!(ax, sR ; markersize = 10, color = :blue)

barplot!(ax_sensors, sensActs; color = sensColors, strokecolor = :black, strokewidth = 1)

barplot!(ax_acts, [1], meanAct; color = :black, strokecolor = :black, strokewidth = 1)
barplot!(ax_acts, [2], meanErr; color = :black, strokecolor = :black, strokewidth = 1)
barplot!(ax_acts, [3], meanTarg; color = :black, strokecolor = :black, strokewidth = 1)


barplot!(ax_effectors, effectorActs; color = :black, strokecolor = :black, strokewidth = 1)

hist!(ax_weights, _weights2)

on(abmobs.model) do m
    autolimits!(ax_weights)
end

############

G = SimpleWeightedDiGraph(link_mat)
pos_x, pos_y = GraphPlot.spring_layout(G)

# Create plot points
edge_x = []
edge_y = []

for edge in Graphs.edges(G)
    push!(edge_x, pos_x[Graphs.src(edge)])
    push!(edge_x, pos_x[Graphs.dst(edge)])
    push!(edge_y, pos_y[Graphs.src(edge)])
    push!(edge_y, pos_y[Graphs.dst(edge)])
end

#  Color node points by the number of connections.
#color_map = [spikes[node] for node in 1:nnodes]

lines!(ax_network, reduce(vcat,edge_x), reduce(vcat,edge_y), color=(:black, .1))

scatter!(ax_network, reduce(vcat, pos_x), reduce(vcat,pos_y); markersize = 20, color = colors)

########
## interactive fig
# fig

#save video
frames = 1:2000
record(fig, "BraitenbergAgent_perturb2.mp4", frames; framerate = 40) do i
    step!(abmobs, 1)
end

#output data
ts_pos_ = DataFrame(ts_pos)
ts_spikes_ = DataFrame(ts_spikes)
ts_acts_ = DataFrame(ts_acts)
ts_heading_ = DataFrame(heading = ts_heading)
ts_sens_ = DataFrame(ts_sens)
ts_eff_ = DataFrame(ts_eff)
ts_hits_ = DataFrame(hits = ts_hits)

using CSV
fill = "_perturb2"
CSV.write("./WallAvoidance/Data/pos" * fill * ".csv", ts_pos_)
CSV.write("./WallAvoidance/Data/spikes" * fill * ".csv", ts_spikes_)
CSV.write("./WallAvoidance/Data/acts" * fill * ".csv", ts_acts_)
CSV.write("./WallAvoidance/Data/heading" * fill * ".csv", ts_heading_)
CSV.write("./WallAvoidance/Data/sens" * fill * ".csv", ts_sens_)
CSV.write("./WallAvoidance/Data/eff" * fill * ".csv", ts_eff_)
CSV.write("./WallAvoidance/Data/hits" * fill * ".csv", ts_hits_)


