#Script containing functions to load and prepare the human behavioural data.

#function which creates new data columns containing a shifted version of a
#variable e.g. previous word reading time.
function shift(data, column, new_column)
    data[new_column] = [missing; data[column][1:end-1]]
end
#prepare the data, i.e. normalisation and creating shifted variables where needed
#eye tracking, N400 and self paced reading data (options et, eeg and spr respectively)
function prep_data(h_data_loc, surp_loc, subj_loc, exp)
    #read in the csv file, the human data is tab separated, the surprisal data comma
    h_data = CSV.read(h_data_loc)
    surp_data = CSV.read(surp_loc)
    subj_data = CSV.read(subj_loc)
    #create an index variable to restore original data order after merging with
    #the surprisal data
    h_data[:idx] = 1:size(h_data,1)
    #select the fixed effects to normalise and create shifted variables if needed
    if exp == "eeg"
        fe_cols = [:word_pos, :nr_char, :log_freq, :baseline, :subtlex_freq]
        # keep track of subjects to keep (reject == false)
        subj_data = subj_data[(subj_data[:experiment] .== "EEG") .& (subj_data[:reject] .== false), :]
    elseif exp == "spr"
        fe_cols = [:word_pos, :nr_char, :log_freq, :prev_nr_char, :prev_log_freq,
                   :prev_RT, :subtlex_freq, :prev_subtlex_freq]
        h_data[:RT] = log.(h_data[:RT])
        shift(h_data, :nr_char, :prev_nr_char)
        shift(h_data, :log_freq, :prev_log_freq)
        shift(h_data, :subtlex_freq, :prev_subtlex_freq)
        shift(h_data, :RT, :prev_RT)
        subj_data = subj_data[(subj_data[:experiment] .== "SPR") .& (subj_data[:reject] .== false), :]
    elseif exp == "et"
        fe_cols = [:word_pos, :nr_char, :log_freq, :prev_nr_char, :prev_log_freq,
                   :subtlex_freq, :prev_subtlex_freq]
        h_data[:RTfirstpass] = log.(h_data[:RTfirstpass])
        shift(h_data, :nr_char, :prev_nr_char)
        shift(h_data, :log_freq, :prev_log_freq)
        shift(h_data, :subtlex_freq, :prev_subtlex_freq)
        subj_data = subj_data[(subj_data[:experiment] .== "ET") .& (subj_data[:reject] .== false), :]
    end
    #select the collumns containing the surprisal data
    surp_cols = [x for x in names(surp_data) if occursin("tf", string(x)) || occursin("gru", string(x))]
    #normalise the fixed effects. Ignore missing values in mean and std
    for fe in fe_cols
        h_data[fe] = ((h_data[fe] .- Statistics.mean(skipmissing(h_data[fe])))
                       / Statistics.std(skipmissing(h_data[fe]))
                      )
    end
    # filter out the datapoints that should be rejected
    reject = []
    ex_next = true
    for word in h_data.word
        r = occursin(",", word) | occursin(".", word)
        reject = [reject; r | ex_next]
        if exp == "eeg"
            ex_next = occursin(".", word)
        else
            ex_next = occursin(".", word) | occursin(",", word)
        end
    end
    # filter out the subjects that should be rejected
    reject_subj = []
    for subj in h_data.subj_nr
        reject_subj = [reject_subj; !(in(subj, subj_data.subj_nr))]
    end
    h_data.reject_word = reject
    h_data.reject_subj = reject_subj

    return h_data, surp_data, surp_cols
end
