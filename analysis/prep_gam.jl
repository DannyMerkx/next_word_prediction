using CategoricalArrays, CSV

function prep_gam(data, n_states, n_reps, method_names)

    n_models = size(data)[1] / (n_states * n_reps)

    #create a repetition variable, basically grouping all save states from a DNN
    #training round as the same repetition
    rep = []
    for x in range(1, n_models)
        for y in range(1, n_reps)
            append!(rep, repeat([1], n_states) * y)
        end
    end
    rep = convert(Array{Int}, rep)
    data[:rep] = rep

    method = []
    factor = []
    f = 0
    for name in method_names
        f += 1
        for y in range(1, stop = n_states * n_reps)
            append!(method, [name])
            append!(factor, f)
        end
    end
    state = []
    for x in range(1, n_models * n_reps)
        for y in range(1, n_states)
            append!(state, y)
        end
    end
    data[:state] = state
    data[:method] = method
    data[:factor] = factor * 1
    #flip the sign of the model surprisal so the plots are ordered from high to low
    #surprisal
    data[:avg_surp] = -data[:avg_surp]

    return(data)
end
