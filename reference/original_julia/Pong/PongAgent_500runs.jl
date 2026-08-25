using JLD2 #SimpleWeightedGraphs, Graphs, JLD2
#import GraphPlot

using Distributed, ClusterManagers
# addprocs(7)

addprocs(SlurmManager(192), N=8, topology=:master_worker)



@everywhere begin
    include("./Pong/PongAgent_Functions.jl");

    # global meanHits_all= []
    # global first100_all = []
    # global last100_all = []
    global learn_on = 1

    function run_model(n)

        global nnodes = 500
        global p_link = .1
        global leak = .25 #
        global leaktype = 1 #1 = leak a percentage, or 2 = leak a constant rate
        global lrate_wmat = .1
        global lrate_targ = .1
        global targ_min = 1.0 
        global movement_amp = 100
        global input_amp = 2.75
        global paddle_radius = 50
        global stim_speed = 5
        global acts_neg = 1

        ####

        global sens_degrees = collect(-90:4:90)
        sensColors= reduce(vcat,[repeat([:red],length(sens_degrees)), repeat([:blue], length(sens_degrees))])

        #sensory_nodes = StatsBase.sample(range(1,nnodes),Weights(repeat([1],nnodes)),Int(round(nnodes*p_link)))
        global input_wmat=zeros((length(sens_degrees),nnodes))
        for row in range(1,size(input_wmat)[1])
            #for col in sensory_nodes
            for col in range(1,nnodes)
                input_wmat[row,col]=StatsBase.sample([0,input_amp],Weights([1-p_link,p_link]))
                #input_wmat[row,col]=StatsBase.sample([0,input_amp],Weights([.8,.2]))
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
                    inhibitory = StatsBase.sample([0,1],Weights([.75,.25]))
                    if inhibitory == 1
                        wmat[][row,col] =  rand(Normal(-1,.1))
                    else
                        wmat[][row,col] =  rand(Normal(0,.2))
                    end
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

        global spikes = Observable(zeros(nnodes))
        global targets = repeat([targ_min],nnodes)
        global acts = zeros(nnodes)
        global inputs = zeros(length(sens_degrees)*2)
        global output_acts = zeros(2)

        global n = 0
        global direction = 1
        global mean_act = 0.0
        global mean_err = 0.0
        global mean_targ = mean(targets)

        global model = ABM(Agent, 
        ContinuousSpace((1000,500), 
        periodic = false); 
        properties = Dict(
            :n => n,
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
                (100.0,250.0),
                π/2,
                sens_degrees,
                []

            ),
            model
        )

        add_agent_pos!(
            Stimulus(
                nextid(model),
                ( 500.0, 250.0 ),

            ),
            model
        )

        ####### stepping 

        nSteps = 100000
        step!(model, dummystep, model_step!, nSteps)

        hits = model[1].hits
        #plot(cumsum(hits))
        meanHits = mean(hits)
        #push!(meanHits_all, meanHits)
        meanHits_first100 = mean(hits[1:100])
        #push!(first100_all, meanHits_first100)
        meanHits_last100 = mean(hits[(end-100):end])
        #push!(last100_all, meanHits_last100)
        return [meanHits, meanHits_first100, meanHits_last100, hits]

    end

end

nRuns = 500
p = pmap(run_model, 1:nRuns; distributed = true, batch_size = 1);

global meanHits_all= []
global first100_all = []
global last100_all = []
global hits_all = []
global first_50 = []
global last_50 = []

for n in 1:length(p)
    push!(meanHits_all, p[n][1])
    push!(first100_all, p[n][2])
    push!(last100_all, p[n][3])
    push!(hits_all, p[n][4])
    push!(first_50, mean(p[n][4][1:50]))
    push!(last_50, mean(p[n][4][(end-49):end]))
end



save_object("meanHits_all_nL.jld2", meanHits_all)
save_object("first100_all_nL.jld2", first100_all)
save_object("last100_all_nL.jld2", last100_all)
save_object("hits_all_nL.jld2", hits_all)

# mean(meanHits_all)
# std(meanHits_all)

# mean(first_50)
# std(first_50)
# mean(first100_all)
# std(first100_all)

# mean(last_50)
# std(last_50)
# mean(last100_all)
# std(last100_all)

# EqualVarianceTTest(convert(AbstractVector{Real}, first_50), convert(AbstractVector{Real}, last_50))


######### reset


