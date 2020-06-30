#This script creates the baseline models and the surprisal models for the
#eye tracking, N400 and self paced reading data (options et, eeg and spr respectively)

#create baseline models (without surprisal as fixed effect)
function baseline_model(data, exp)
    if exp == "eeg"
        #formula for the mixed model
        formula = @formula(N400 ~ nr_char + subtlex_freq + word_pos + baseline +
                           (nr_char * subtlex_freq) +
                           (nr_char * word_pos) +
                           (nr_char * baseline) +
                           (subtlex_freq * word_pos) +
                           (subtlex_freq * baseline) +
                           (word_pos * baseline) +
                           (1 | subj_nr) +
                           (0 + nr_char | subj_nr) +
                           (0 + subtlex_freq | subj_nr) +
                           (0 + word_pos | subj_nr) +
                           (0 + baseline | subj_nr) +
                           (1 | item)
                           )
        #list of all the columns that are needed for the MLM
        cols = [:N400, :nr_char, :subtlex_freq, :word_pos, :baseline, :subj_nr,
                :item]
    elseif exp == "spr"
        #formula for the mixed model
        formula = @formula(RT ~ nr_char + subtlex_freq + word_pos +
                           prev_nr_char + prev_subtlex_freq + prev_RT +
                           (nr_char * subtlex_freq) +
                           (nr_char * word_pos) +
                           (nr_char * prev_nr_char) +
                           (nr_char * prev_subtlex_freq) +
                           (nr_char * prev_RT) +
                           (subtlex_freq * word_pos) +
                           (subtlex_freq * prev_nr_char) +
                           (subtlex_freq * prev_subtlex_freq) +
                           (subtlex_freq * prev_RT) +
                           (word_pos * prev_nr_char) +
                           (word_pos * prev_subtlex_freq) +
                           (word_pos * prev_RT) +
                           (prev_nr_char * prev_subtlex_freq) +
                           (prev_nr_char * prev_RT) +
                           (prev_subtlex_freq * prev_RT) +
                           (1 | subj_nr) +
                           (0 + nr_char | subj_nr) +
                           (0 + subtlex_freq | subj_nr) +
                           (0 + word_pos | subj_nr) +
                           (0 + prev_nr_char | subj_nr) +
                           (0 + prev_subtlex_freq | subj_nr) +
                           (0 + prev_RT | subj_nr) +
                           (1 | item)
                           )
        #list of all the columns that are needed for the MLM
        cols = [:RT, :nr_char, :subtlex_freq, :word_pos, :prev_nr_char,
                :prev_subtlex_freq, :prev_RT, :subj_nr, :item]
    elseif exp == "et"
        #formula for the mixed model
        formula = @formula(RTfirstpass ~ nr_char + subtlex_freq + word_pos +
                           prev_nr_char + prev_subtlex_freq + prevfix +
                           (nr_char * subtlex_freq) +
                           (nr_char * word_pos) +
                           (nr_char * prev_nr_char) +
                           (nr_char * prev_subtlex_freq) +
                           (nr_char * prevfix) +
                           (subtlex_freq * word_pos) +
                           (subtlex_freq * prev_nr_char) +
                           (subtlex_freq * prev_subtlex_freq) +
                           (subtlex_freq * prevfix) +
                           (word_pos * prev_nr_char) +
                           (word_pos * prev_subtlex_freq) +
                           (word_pos * prevfix) +
                           (prev_nr_char * prev_subtlex_freq) +
                           (prev_nr_char * prevfix) +
                           (prev_subtlex_freq * prevfix) +
                           (1 | subj_nr) +
                           (0 + nr_char | subj_nr) +
                           (0 + subtlex_freq | subj_nr) +
                           (0 + word_pos | subj_nr) +
                           (0 + prev_nr_char | subj_nr) +
                           (0 + prev_subtlex_freq | subj_nr) +
                           (0 + prevfix | subj_nr) +
                           (1 | item)
                           )
        #list of all the columns that are needed for the MLM
        cols = [:RTfirstpass, :nr_char, :subtlex_freq, :word_pos, :prev_nr_char,
                :prev_subtlex_freq, :prevfix, :subj_nr, :item]
    end
    #fit the model, remove missing valuas and rows marked for rejection
    model = fit(LinearMixedModel, formula,
                data[completecases(data) .& (data[:reject_data] .== false) .& (data[:reject_word] .== false) .& (data[:reject_subj] .== false), :][cols],
                REML = false)
    #extract the baseline deviance
    return(model, deviance(model))
end

function join_data(h_data, surp_data, surp)
    #join the subject data with the current surprisal data
    h_data = join(surp_data[[surp, :item]], h_data, on = :item, kind = :inner,
                  makeunique = true)
    #rename the surprisal column to the name used in the LME formula
    rename!(h_data, (surp => :surp))
    #sort the data back into the original order
    h_data = sort!(h_data, :idx)
    #extract the average surprisal for the current DNN model and normalise the
    #data
    avg_surp = mean(h_data[completecases(h_data) .& (h_data[:reject_data] .== false) .& (h_data[:reject_word] .== false) .& (h_data[:reject_subj] .== false), :surp])
    h_data[:surp] = ((h_data[:surp] .- Statistics.mean(skipmissing(h_data[:surp])))
                      / Statistics.std(skipmissing(h_data[:surp]))
                     )
    return h_data, avg_surp
end
#create models with surprisal as fixed effect.
function surp_model(h_data, surp_data, exp, surp)
    if exp == "eeg"
           #formula for the mixed model

           formula = @formula(N400 ~ surp + nr_char + subtlex_freq + word_pos +
                              baseline +
                              (nr_char * subtlex_freq) +
                              (nr_char * word_pos) +
                              (nr_char * baseline) +
                              (subtlex_freq * word_pos) +
                              (subtlex_freq * baseline) +
                              (word_pos * baseline) +
                              (1 | subj_nr) +
                              (0 + surp | subj_nr) +
                              (0 + nr_char | subj_nr) +
                              (0 + subtlex_freq | subj_nr) +
                              (0 + word_pos | subj_nr) +
                              (0 + baseline | subj_nr) +
                              (1 | item))
           h_data, avg_surp = join_data(h_data, surp_data, surp)
           #list of all the columns that are needed for the MLM
           cols = [:N400, :surp, :nr_char, :subtlex_freq, :word_pos, :baseline,
                   :subj_nr, :item]
    elseif exp == "spr"
        #formula for the mixed model
        formula = @formula(RT ~ surp +  prev_surp + nr_char + log_freq +
                           word_pos + prev_nr_char + prev_log_freq + prev_RT +
                           (nr_char * log_freq) +
                           (nr_char * word_pos) +
                           (nr_char * prev_nr_char) +
                           (nr_char * prev_log_freq) +
                           (nr_char * prev_RT) +
                           (log_freq * word_pos) +
                           (log_freq * prev_nr_char) +
                           (log_freq * prev_log_freq) +
                           (log_freq * prev_RT) +
                           (word_pos * prev_nr_char) +
                           (word_pos * prev_log_freq) +
                           (word_pos * prev_RT) +
                           (prev_nr_char * prev_log_freq) +
                           (prev_nr_char * prev_RT) +
                           (prev_log_freq * prev_RT) +
                           (1 | subj_nr) +
                           (0 + surp | subj_nr) +
                           (0 + prev_surp | subj_nr) +
                           (0 + nr_char | subj_nr) +
                           (0 + log_freq | subj_nr) +
                           (0 + word_pos | subj_nr) +
                           (0 + prev_nr_char | subj_nr) +
                           (0 + prev_log_freq | subj_nr) +
                           (0 + prev_RT | subj_nr) +
                           (1 | item))
        h_data, avg_surp = join_data(h_data, surp_data, surp)
        # create a previous surprisal variable
        h_data[:prev_surp] = [missing; h_data[:surp][1:end-1]]
        #list of all the columns that are needed for the MLM
        cols = [:RT, :surp, :prev_surp, :nr_char, :log_freq, :word_pos,
                :prev_nr_char, :prev_log_freq, :prev_RT, :subj_nr, :item]
    elseif exp == "et"
        #formula for the mixed model
        formula = @formula(RTfirstpass ~ surp + prev_surp + nr_char +
                           subtlex_freq + word_pos + prev_nr_char +
                           prev_subtlex_freq + prevfix +
                           (nr_char * subtlex_freq) +
                           (nr_char * word_pos) +
                           (nr_char * prev_nr_char) +
                           (nr_char * prev_subtlex_freq) +
                           (nr_char * prevfix) +
                           (subtlex_freq * word_pos) +
                           (subtlex_freq * prev_nr_char) +
                           (subtlex_freq * prev_subtlex_freq) +
                           (subtlex_freq * prevfix) +
                           (word_pos * prev_nr_char) +
                           (word_pos * prev_subtlex_freq) +
                           (word_pos * prevfix) +
                           (prev_nr_char * prev_subtlex_freq) +
                           (prev_nr_char * prevfix) +
                           (prev_subtlex_freq * prevfix) +
                           (1 | subj_nr) +
                           (0 + surp | subj_nr) +
                           (0 + prev_surp | subj_nr) +
                           (0 + nr_char | subj_nr) +
                           (0 + subtlex_freq | subj_nr) +
                           (0 + word_pos | subj_nr) +
                           (0 + prev_nr_char | subj_nr) +
                           (0 + prev_subtlex_freq | subj_nr) +
                           (0 + prevfix | subj_nr) +
                           (1 | item))
        h_data, avg_surp = join_data(h_data, surp_data, surp)
        # create a previous surprisal variable
        h_data[:prev_surp] = [missing; h_data[:surp][1:end-1]]
        #list of all the columns that are needed for the MLM
        cols = [:RTfirstpass, :surp, :prev_surp, :nr_char, :subtlex_freq,
                :word_pos, :prev_nr_char, :prev_subtlex_freq, :prevfix,
                :subj_nr, :item]
    end
    #fit the model, remove missing values and rows marked for rejection
    model = fit(LinearMixedModel, formula,
                h_data[completecases(h_data) .& (h_data[:reject_data] .== false) .& (h_data[:reject_word] .== false) .& (h_data[:reject_subj] .== false), :][cols],
                REML = false)
    #extract the deviance of the model and the coefficient of the surprisal
    #fixed effect
    dev = deviance(model)
    surp_coef = coeftable(model).cols[1][2]
    return model, [dev, surp_coef, avg_surp]
end
