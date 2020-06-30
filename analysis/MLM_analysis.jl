using MixedModels, CSV, DataFrames, Statistics, StatsBase
include("prep_data.jl")
include("MLM.jl")
# locations of the datafiles
EEG_loc = "/home/danny/Documents/databases/next_word_prediction/data/data_EEG.csv"
SPR_loc = "/home/danny/Documents/databases/next_word_prediction/data/data_SPR.csv"
ET_loc = "/home/danny/Documents/databases/next_word_prediction/data/data_ET.csv"
subj_loc = "/home/danny/Documents/databases/next_word_prediction/data/subjects.csv"
surp_loc = "/home/danny/Documents/databases/next_word_prediction/surprisal_data/surprisal.csv"

# eeg analysis
eeg_data, surp_data, surp_cols = prep_data(EEG_loc, surp_loc, subj_loc, "eeg")
model, base = baseline_model(eeg_data, "eeg")

eeg_results = []
for m in surp_cols
    model, results = @time surp_model(eeg_data, surp_data, "eeg", m)
    print(string(m, ' ', -(base - results[1]) * sign(results[2]), '\n'))
    global eeg_results = [eeg_results; (-(base - results[1]) * sign(results[2]),
                          string(m), results[3])]
end
eeg_results = DataFrame([[x[1] for x in eeg_results],[x[2] for x in eeg_results],
          [x[3] for x in eeg_results]], [:fit_score, :model_name, :avg_surp])
CSV.write("/home/danny/Documents/databases/next_word_prediction/MLM_results/results_eeg.csv",
          DataFrame(eeg_results), writeheader = true)

# eye tracking analysis
et_data, surp_data, surp_cols = prep_data(ET_loc, surp_loc, subj_loc, "et")
model, base = @time baseline_model(et_data, "et")

et_results = []
for m in surp_cols
  model, results = @time surp_model(et_data, surp_data, "et", m)
  print(string(m, ' ', (base - results[1]) * sign(results[2]), '\n'))
  global et_results = [et_results; ((base - results[1]) * sign(results[2]),
                        string(m), results[3])]
end
et_results = DataFrame([[x[1] for x in et_results],[x[2] for x in et_results],
        [x[3] for x in et_results]], [:fit_score, :model_name, :avg_surp])
CSV.write("/home/danny/Documents/databases/next_word_prediction/MLM_results/results_et.csv",
        DataFrame(et_results), writeheader = true)

# self paced reading analysis
spr_data, surp_data, surp_cols = prep_data(SPR_loc, surp_loc, subj_loc, "spr")
model, base = @time baseline_model(spr_data, "spr")

spr_results = []
for m in surp_cols
    model, results = @time surp_model(spr_data, surp_data, "spr", m)
    print(string(m, ' ', (base - results[1]) * sign(results[2]), '\n'))
    global spr_results = [spr_results; ((base - results[1]) * sign(results[2]),
                          string(m), results[3])]
end

spr_results = DataFrame([[x[1] for x in spr_results],[x[2] for x in spr_results],
          [x[3] for x in spr_results]], [:fit_score, :model_name, :avg_surp])
CSV.write("/home/danny/Documents/databases/next_word_prediction/MLM_results/results_spr.csv",
          DataFrame(spr_results), writeheader = true)

# self paced reading ET sentences only
SPR_loc = "/home/danny/Documents/databases/next_word_prediction/data/data_SPR_ET.csv"

spr_data, surp_data, surp_cols = prep_data(SPR_loc, surp_loc, subj_loc, "spr")
model, base = @time baseline_model(spr_data, "spr")

spr_results = []
for m in surp_cols
    model, results = @time surp_model(spr_data, surp_data, "spr", m)
    print(string(m, ' ', (base - results[1]) * sign(results[2]), '\n'))
    global spr_results = [spr_results; ((base - results[1]) * sign(results[2]),
                          string(m), results[3])]
end

spr_results = DataFrame([[x[1] for x in spr_results],[x[2] for x in spr_results],
          [x[3] for x in spr_results]], [:fit_score, :model_name, :avg_surp])
CSV.write("/home/danny/Documents/databases/next_word_prediction/MLM_results/results_spr_et.csv",
          DataFrame(spr_results), writeheader = true)
# self paced reading, everything but ET sentences
SPR_loc = "/home/danny/Documents/databases/next_word_prediction/data/data_SPR_ET2.csv"

spr_data, surp_data, surp_cols = prep_data(SPR_loc, surp_loc, subj_loc, "spr")
model, base = @time baseline_model(spr_data, "spr")

spr_results = []
for m in surp_cols
    model, results = @time surp_model(spr_data, surp_data, "spr", m)
    print(string(m, ' ', (base - results[1]) * sign(results[2]), '\n'))
    global spr_results = [spr_results; ((base - results[1]) * sign(results[2]),
                          string(m), results[3])]
end

spr_results = DataFrame([[x[1] for x in spr_results],[x[2] for x in spr_results],
          [x[3] for x in spr_results]], [:fit_score, :model_name, :avg_surp])
CSV.write("/home/danny/Documents/databases/next_word_prediction/MLM_results/results_spr_et2.csv",
          DataFrame(spr_results), writeheader = true)
