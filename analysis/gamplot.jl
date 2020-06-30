#Script which prepares data for GAM in Julia and performs GAM in R
using RCall, CategoricalArrays, CSV
include("prep_gam.jl")
R"library(mgcv)"
R"library(itsadug)"
R"library(ggplot2)"
R"library(mgcViz)"
R"library(voxel)"
# load the data
eeg_data = CSV.read("/home/danny/Documents/databases/next_word_prediction/MLM_results/results_eeg_subtlex.csv",
                    header = true)
et_data = CSV.read("/home/danny/Documents/databases/next_word_prediction/MLM_results/results_et_subtlex.csv",
                   header = true)
spr_data = CSV.read("/home/danny/Documents/databases/next_word_prediction/MLM_results/results_spr_subtlex.csv",
                    header = true)

#the number of save states per training run, number of training runs per model
#and the number of models
n_states = 10
n_reps = 8
#create a column with the model name e.g. tf_one_layer, gru_one_layer etc.
method_names = ["gru2", "gru", "gru_tf",  "tf2", "tf4", "tf"]
# prepare the data
eeg_data = prep_gam(eeg_data, n_states, n_reps, method_names)
et_data = prep_gam(et_data, n_states, n_reps, method_names)
spr_data = prep_gam(spr_data, n_states, n_reps, method_names)
# remove negative scores from the data used in the GAMs
eeg_data_gam = eeg_data[.!(eeg_data[:fit_score] .< 0), :]
et_data_gam = et_data[.!(et_data[:fit_score] .< 0), :]
spr_data_gam = spr_data[.!(spr_data[:fit_score] .< 0), :]

#conversion from julia categoricalarrays to R factors led to malformed factors
#so I perform this last preprocessing step in R
@rput(eeg_data)
@rput(et_data)
@rput(spr_data)
@rput(eeg_data_gam)
@rput(et_data_gam)
@rput(spr_data_gam)
############################ main gam models ###################################
# create unordered factors for the training repetition and the model type
R"eeg_data_gam$factor <- factor(eeg_data_gam$factor, ordered = FALSE)"
R"eeg_data_gam$rep <- factor(eeg_data_gam$rep, ordered = FALSE)"

R"et_data_gam$factor <- factor(et_data_gam$factor, ordered = FALSE)"
R"et_data_gam$rep <- factor(et_data_gam$rep, ordered = FALSE)"

R"spr_data_gam$factor <- factor(spr_data_gam$factor, ordered = FALSE)"
R"spr_data_gam$rep <- factor(spr_data_gam$rep, ordered = FALSE)"

# Fit the GAMs
R"eeg_model <- gam(fit_score ~ factor + s(avg_surp, by = factor, k = 30) +
                   s(avg_surp, rep, bs = 'fs', k = 2),
                   data = eeg_data_gam[eeg_data_gam$method == 'gru' |
                                       eeg_data_gam$method == 'gru2' |
                                       eeg_data_gam$method == 'tf' |
                                       eeg_data_gam$method == 'tf2',],
                    method = 'ML'
                    )"
R"gam.check(eeg_model)"

R"et_model <- gam(fit_score ~ factor + s(avg_surp, by = factor, k = 30) +
                  s(avg_surp, rep, bs = 'fs', k = 3),
                  data = et_data_gam[et_data_gam$method == 'gru' |
                                     et_data_gam$method == 'gru2' |
                                     et_data_gam$method == 'tf' |
                                     et_data_gam$method == 'tf2',],
                  method = 'ML'
                  )"
R"gam.check(et_model)"

R"spr_model <- gam(fit_score ~ factor + s(avg_surp, by = factor, k = 30) +
                   s(avg_surp, rep, bs = 'fs', k = 3),
                   data = spr_data_gam[spr_data_gam$method == 'tf' |
                                       spr_data_gam$method == 'tf2' |
                                       spr_data_gam$method == 'gru' |
                                       spr_data_gam$method == 'gru2',],
                   method = 'ML'
                   )"
R"gam.check(spr_model)"

############################## Plot GAM smooths ################################
R"jpeg('/home/danny/Pictures/gam.jpg', res = 150, width = 1400, height = 1000)"
# plot data including negatives in scatterplot
R"par(mfrow=c(2,3), mai = c(0.4, 0.4, 0.2, 0.1), cex = '1')"
R"plot(eeg_data[eeg_data$method == 'gru' | eeg_data$method == 'tf' |
                eeg_data$method == 'gru2' | eeg_data$method == 'tf2',]$avg_surp,
       eeg_data[eeg_data$method == 'gru' | eeg_data$method == 'tf' |
                eeg_data$method == 'gru2' | eeg_data$method == 'tf2',]$fit_score,
       main='N400 size', col = c(rep('red', 80),rep('green', 80),
       rep('cyan', 80),rep('purple', 80) ), xlab='', xlim = c(-8, -4.5),
       ylab='goodness-of-fit', pch = c(rep(1, 80),rep(2, 80),
                                       rep(3, 80),rep(4, 80)
                                       )
       )

grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)

legend('topleft', bg = 'white',
       legend = c('2L GRU', 'GRU', '2L Transformer','Transformer'),
       col=c('red', 'green', 'cyan', 'purple'), pch = c(1,2,3,4)
       )"

R"plot(et_data[et_data$method == 'gru' | et_data$method == 'tf' |
               et_data$method == 'gru2' | et_data$method == 'tf2',]$avg_surp,
       et_data[et_data$method == 'gru' | et_data$method == 'tf' |
               et_data$method == 'gru2' | et_data$method == 'tf2',]$fit_score,
       main='Gaze duration', col = c(rep('red', 80),rep('green', 80),
       rep('cyan', 80),rep('purple', 80) ), xlab='', xlim = c(-8.5, -5),
       ylab='', pch = c(rep(1, 80),rep(2, 80), rep(3, 80),rep(4, 80))
       )

grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)"

R"plot(spr_data[spr_data$method == 'gru' | spr_data$method == 'tf' |
                spr_data$method == 'gru2' | spr_data$method == 'tf2',]$avg_surp,
       spr_data[spr_data$method == 'gru' | spr_data$method == 'tf' |
                spr_data$method == 'gru2' | spr_data$method == 'tf2',]$fit_score,
       main='Self paced reading time', col = c(rep('red', 80),rep('green', 80),
       rep('cyan', 80),rep('purple', 80) ), xlab='', xlim = c(-8, -4.5),
       ylab='', pch = c(rep(1, 56),rep(2, 80), rep(3, 80),rep(4, 80))
       )

grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)"
#plot the GAMs
R"p <- plot_smooth(eeg_model, view= 'avg_surp', plot_all = 'factor', se = TRUE,
                   shade = TRUE, xlab = '', ylab = 'goodness-of-fit',
                   xlim = c(-8, -4.5), ylim = c(-30, 140), lty = c(1,2,3,6),
                   lwd = 2, legend_plot_all = FALSE, rug = FALSE, alpha = 0.4,
                   hide.label = T
                   )

grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)

box(lty = 'solid', col = 'black')

legend('topleft', bg = 'white',
       legend = c('2L GRU','GRU', '2L Transformer', 'Transformer'),
       col=c('red', 'green', 'cyan', 'purple'), lty = c(1,2,3,6), lwd = 2
       )"

R"p <- plot_smooth(et_model, view= 'avg_surp', plot_all = 'factor', se = TRUE,
                   shade = TRUE, xlim = c(-8.5, -5), xlab = 'average surprisal',
                   ylab = '', ylim = c(0, 200), lty = c(1,2,3,6), lwd = 2,
                   legend_plot_all = FALSE, rug = FALSE, alpha = 0.4,
                   hide.label = T
                   )

grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)

box(lty = 'solid', col = 'black')"

R"p <- plot_smooth(spr_model, view= 'avg_surp', plot_all = 'factor', se = TRUE,
                   shade = TRUE, xlim = c(-8, -4.5), xlab = '', ylab = '',
                   ylim = c(-30, 240), lty = c(1,2,3,6), lwd = 2,
                   legend_plot_all = FALSE, rug = FALSE, alpha = 0.4,
                   hide.label = T
                   )

grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)

box(lty = 'solid', col = 'black')"

R"dev.off()"

##############plot main difference curves of interest ##########################
R"jpeg('/home/danny/Pictures/diff.jpg', res = 150, width = 1400, height = 500)"

R"par(mfrow=c(1,3), mai = c(0.4, 0.4, 0.2, 0.1), cex = '1')"
R"p <- plot_diff(eeg_model, view = 'avg_surp', comp = list(factor=c(1,2)),
                 col = '#FF0000', shade = TRUE, col.dif = '#FF0000',
                 ylim = c(-15, 15), lwd = 3, alpha = 0.4,
                 ylab = 'Estimated difference in goodness-of-fit score',
                 main = 'N400', hide.label = TRUE, n.grid = 1000
                 )

grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)

box(lty = 'solid', col = 'black')"

R"p <- plot_diff(eeg_model, view = 'avg_surp', comp = list(factor=c(6,4)),
                 add = TRUE, col = '#00FF00', shade = TRUE, col.dif = '#00FF00',
                 lty = 2, lwd = 3, alpha = 0.4, n.grid = 1000
                 )"

R"p <- plot_diff(eeg_model, view = 'avg_surp', comp = list(factor=c(2,4)),
                 add = TRUE, col = '#0000FF', shade = TRUE, col.dif = '#0000FF',
                 lty = 3, lwd = 3, alpha = 0.4, n.grid = 1000
                 )
abline(h = 0)
"

R"p <- plot_diff(et_model, view = 'avg_surp', comp = list(factor=c(1,2)),
                 col = '#FF0000', shade = TRUE, col.dif = '#FF0000',
                 ylim = c(-30, 40), lwd = 3, alpha = 0.4,
                 xlab = 'average surprisal', ylab = '', main = 'Gaze duration',
                 hide.label = TRUE, n.grid = 1000
                 )

grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)

box(lty = 'solid', col = 'black')"

R"p <- plot_diff(et_model, view = 'avg_surp', comp = list(factor=c(6,4)),
                 add = TRUE, col = '#00FF00', shade = TRUE,
                 col.dif = '#00FF00', lty = 2, lwd = 3, alpha = 0.4,
                 n.grid = 1000)"

R"p <- plot_diff(et_model, view = 'avg_surp', comp = list(factor=c(2,4)),
                 add = TRUE, col = '#0000FF', shade = TRUE, col.dif = '#0000FF',
                 lty = 3, lwd = 3, alpha = 0.4, n.grid = 1000
                 )
abline(h = 0)

legend('top', bg = 'white', legend = c('2L GRU - GRU',
                                       'Transformer - 2L Transformer',
                                       'GRU - 2L Transformer'
                                       ),
       col=c('#FF0000', '#00FF00', '#0000FF'), lty = c(1,2,3), lwd = 2,
       cex = 0.85
       )"

R"p <- plot_diff(spr_model, view = 'avg_surp', comp = list(factor=c(1,2)),
                 col = '#FF0000', shade = TRUE, col.dif = '#FF0000',
                 ylim = c(-30, 30), lwd = 3, alpha = 0.4, ylab = '',
                 main = 'Self paced reading', hide.label = TRUE, n.grid = 1000)

grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)

box(lty = 'solid', col = 'black')"

R"p <- plot_diff(spr_model, view = 'avg_surp', comp = list(factor=c(6,4)),
                 add = TRUE, col = '#00FF00', shade = TRUE, col.dif = '#00FF00',
                 lty = 2, lwd = 3, alpha = 0.4, n.grid = 1000
                 )"

R"p <- plot_diff(spr_model, view = 'avg_surp', comp = list(factor=c(2,4)),
                 add = TRUE, col = '#0000FF', shade = TRUE, col.dif = '#0000FF',
                 lty = 3, lwd = 3, alpha = 0.4, n.grid = 1000
                 )
abline(h = 0)"

R"dev.off()"

######################### plot the remaining diff curves #######################
R"jpeg('/home/danny/Pictures/diff2.jpg', res = 150, width = 1400, height = 500)"

R"par(mfrow=c(1,3), mai = c(0.4, 0.4, 0.2, 0.1), cex = '1')"
R"p <- plot_diff(eeg_model, view = 'avg_surp', comp = list(factor=c(1,6)),
                 col = '#FF0000', shade = TRUE, col.dif = '#FF0000',
                 ylim = c(-15, 15), lwd = 3, alpha = 0.4,
                 ylab = 'Estimated difference in goodness-of-fit score',
                 main = 'N400', hide.label = TRUE, n.grid = 1000)
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)
box(lty = 'solid', col = 'black')"
R"p <- plot_diff(eeg_model, view = 'avg_surp', comp = list(factor=c(2,6)),
                 add = TRUE, col = '#00FF00', shade = TRUE,
                 col.dif = '#00FF00', lty = 2, lwd = 3, alpha = 0.4,
                 n.grid = 1000)"
R"p <- plot_diff(eeg_model, view = 'avg_surp', comp = list(factor=c(1,4)),
                 add = TRUE, col = '#0000FF', shade = TRUE, col.dif = '#0000FF',
                 lty = 3, lwd = 3, alpha = 0.4, n.grid = 1000)
abline(h = 0)
"

R"p <- plot_diff(et_model, view = 'avg_surp', comp = list(factor=c(1,6)),
                 col = '#FF0000', shade = TRUE, col.dif = '#FF0000',
                 ylim = c(-30, 40), lwd = 3, alpha = 0.4,
                 xlab = 'average surprisal', ylab = '', main = 'Gaze duration',
                 hide.label = TRUE, n.grid = 1000)
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)
box(lty = 'solid', col = 'black')"
R"p <- plot_diff(et_model, view = 'avg_surp', comp = list(factor=c(2,6)),
                 add = TRUE, col = '#00FF00', shade = TRUE,
                 col.dif = '#00FF00', lty = 2, lwd = 3, alpha = 0.4,
                 n.grid = 1000)"
R"p <- plot_diff(et_model, view = 'avg_surp', comp = list(factor=c(1,4)),
                 add = TRUE, col = '#0000FF', shade = TRUE, col.dif = '#0000FF',
                 lty = 3, lwd = 3, alpha = 0.4, n.grid = 1000)
abline(h = 0)
legend('top', bg = 'white', legend = c('2L GRU - Transformer',
                                       'GRU - Transformer',
                                       '2L GRU - 2L Transformer'
                                       ),
       col=c('#FF0000', '#00FF00', '#0000FF'), lty = c(1,2,3), lwd = 2,
       cex = 0.85
       )
"

R"p <- plot_diff(spr_model, view = 'avg_surp', comp = list(factor=c(1,6)),
                 col = '#FF0000', shade = TRUE, col.dif = '#FF0000',
                 ylim = c(-30, 30), lwd = 3, alpha = 0.4, ylab = '',
                 main = 'Self paced reading', hide.label = TRUE, n.grid = 1000)
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)
box(lty = 'solid', col = 'black')"
R"p <- plot_diff(spr_model, view = 'avg_surp', comp = list(factor=c(2,6)),
                 add = TRUE, col = '#00FF00', shade = TRUE,
                 col.dif = '#00FF00', lty = 2, lwd = 3, alpha = 0.4,
                 n.grid = 1000)"
R"p <- plot_diff(spr_model, view = 'avg_surp', comp = list(factor=c(1,4)),
                 add = TRUE, col = '#0000FF', shade = TRUE, col.dif = '#0000FF',
                 lty = 3, lwd = 3, alpha = 0.4, n.grid = 1000)
abline(h = 0)"

R"dev.off()"

################## fit 4l transformer and attention gru gams ###################

R"eeg_model <- gam(fit_score ~ factor + s(avg_surp, by = factor, k = 30) +
                   s(avg_surp, rep, bs = 'fs', k = 2),
                   data = eeg_data_gam[eeg_data_gam$method == 'tf4' |
                                       eeg_data_gam$method == 'tf2',],
                   method = 'ML')"
R"gam.check(eeg_model)"

R"et_model <- gam(fit_score ~ factor + s(avg_surp, by = factor, k = 30) +
                  s(avg_surp, rep, bs = 'fs', k = 3),
                  data = et_data_gam[et_data_gam$method == 'tf4' |
                                     et_data_gam$method == 'tf2',],
                  method = 'ML')"
R"gam.check(et_model)"

R"spr_model <- gam(fit_score ~ factor + s(avg_surp, by = factor, k = 30) +
                   s(avg_surp, rep, bs = 'fs', k = 3),
                   data = spr_data_gam[spr_data_gam$method == 'tf4' |
                                       spr_data_gam$method == 'tf2',],
                   method = 'ML')"
R"gam.check(spr_model)"

R"eeg_model2 <- gam(fit_score ~ factor + s(avg_surp, by = factor, k = 50) +
                    s(avg_surp, rep, bs = 'fs', k = 2),
                    data = eeg_data_gam[eeg_data_gam$method == 'gru' |
                                        eeg_data_gam$method == 'gru_tf',],
                    method = 'ML')"
R"gam.check(eeg_model)"

R"et_model2 <- gam(fit_score ~ factor + s(avg_surp, by = factor, k = 30) +
                   s(avg_surp, rep, bs = 'fs', k = 3),
                   data = et_data_gam[et_data_gam$method == 'gru' |
                                      et_data_gam$method == 'gru_tf',],
                   method = 'ML')"
R"gam.check(et_model)"

R"spr_model2 <- gam(fit_score ~ factor + s(avg_surp, by = factor, k = 30) +
                    s(avg_surp, rep, bs = 'fs', k = 3),
                    data = spr_data_gam[spr_data_gam$method == 'gru' |
                                        spr_data_gam$method == 'gru_tf',],
                    method = 'ML')"
R"gam.check(spr_model)"

####################### plot gams and diff curves ##############################

R"jpeg('/home/danny/Pictures/gam_tf.jpg', res = 150, width = 1400,
       height = 1000)"

R"par(mfrow=c(2,3), mai = c(0.4, 0.4, 0.2, 0.1), cex = '1')"

R"p <- plot_smooth(eeg_model, view= 'avg_surp', plot_all = 'factor', se = TRUE,
                   shade = TRUE, main='N400 size', xlab = '',
                   ylab = 'goodness-of-fit', ylim = c(0, 110), lty = c(1,2,3,6),
                   lwd = 2, legend_plot_all = FALSE, rug = FALSE, alpha = 0.4,
                   hide.label = T)
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)
box(lty = 'solid', col = 'black')
legend('topleft', bg = 'white', legend = c('2L Transformer','4L Transformer'),
       col=c('red', 'cyan'), lty = c(1,2,3,6), lwd = 2)"

R"p <- plot_smooth(et_model, view= 'avg_surp', plot_all = 'factor', se = TRUE,
                   shade = TRUE, main='Gaze duration',
                   xlab = 'average surprisal', ylab = '', ylim = c(20, 180),
                   lty = c(1,2,3,6), lwd = 2, legend_plot_all = FALSE,
                   rug = FALSE, alpha = 0.4, hide.label = T)
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)
box(lty = 'solid', col = 'black')"

R"p <- plot_smooth(spr_model, view= 'avg_surp', plot_all = 'factor', se = TRUE,
                   shade = TRUE, main='Self paced reading time', xlab = '',
                   ylab = '', ylim = c(30, 230), lty = c(1,2,3,6), lwd = 2,
                   legend_plot_all = FALSE, rug = FALSE, alpha = 0.4,
                   hide.label = T)
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)
box(lty = 'solid', col = 'black')"

R"p <- plot_diff(eeg_model, view = 'avg_surp', comp = list(factor=c(4,5)),
                 shade = TRUE, col.dif = '#FF0000', ylim = c(-60, 60), lwd = 3,
                 alpha = 0.4, ylab = '', main = '', hide.label = TRUE,
                 n.grid = 1000)
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)
box(lty = 'solid', col = 'black')
abline(h = 0)"

R"p <- plot_diff(et_model, view = 'avg_surp', comp = list(factor=c(4,5)),
                 shade = TRUE, col.dif = '#FF0000', ylim = c(-60, 60), lwd = 3,
                 alpha = 0.4, ylab = '', main = '', hide.label = TRUE,
                 n.grid = 1000)
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)
box(lty = 'solid', col = 'black')
abline(h = 0)"

R"p <- plot_diff(spr_model, view = 'avg_surp', comp = list(factor=c(4,5)),
                 shade = TRUE, col.dif = '#FF0000', ylim = c(-60, 60), lwd = 3,
                 alpha = 0.4, ylab = '', main = '', hide.label = TRUE,
                 n.grid = 1000)
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)
box(lty = 'solid', col = 'black')
abline(h = 0)"

R"dev.off()"

R"jpeg('/home/danny/Pictures/gru_tf.jpg', res = 150, width = 1400,
       height = 1000)"

R"par(mfrow=c(2,3), mai = c(0.4, 0.4, 0.2, 0.1), cex = '1')"

R"p <- plot_smooth(eeg_model2, view= 'avg_surp', plot_all = 'factor', se = TRUE,
                   shade = TRUE, xlab = '', ylab = 'goodness-of-fit',
                   ylim = c(0, 100), lty = c(1,2,3,6), lwd = 2,
                   legend_plot_all = FALSE, rug = FALSE, alpha = 0.4,
                   hide.label = T)
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)
box(lty = 'solid', col = 'black')
legend('topleft', bg = 'white', legend = c('GRU','GRU Transformer'),
       col=c('red', 'cyan'), lty = c(1,2,3,6), lwd = 2)"

R"p <- plot_smooth(et_model2, view= 'avg_surp', plot_all = 'factor', se = TRUE,
                   shade = TRUE, xlab = 'average surprisal', ylab = '',
                   ylim = c(50, 200), lty = c(1,2,3,6), lwd = 2,
                   legend_plot_all = FALSE, rug = FALSE, alpha = 0.4,
                   hide.label = T)
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)
box(lty = 'solid', col = 'black')"

R"p <- plot_smooth(spr_model2, view= 'avg_surp', plot_all = 'factor',
                   se = TRUE, shade = TRUE, xlab = '', ylab = '',
                   ylim = c(0, 220), lty = c(1,2,3,6), lwd = 2,
                   legend_plot_all = FALSE, rug = FALSE, alpha = 0.4,
                   hide.label = T)
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)
box(lty = 'solid', col = 'black')"

R"p <- plot_diff(eeg_model2, view = 'avg_surp', comp = list(factor=c(2,3)),
                 shade = TRUE, col.dif = '#FF0000', ylim = c(-60, 60), lwd = 3,
                 alpha = 0.4, ylab = '', main = '', hide.label = TRUE,
                 n.grid = 1000)
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)
box(lty = 'solid', col = 'black')
abline(h = 0)"

R"p <- plot_diff(et_model2, view = 'avg_surp', comp = list(factor=c(2,3)),
                 shade = TRUE, col.dif = '#FF0000', ylim = c(-60, 60), lwd = 3,
                 alpha = 0.4, ylab = '', main = '', hide.label = TRUE,
                 n.grid = 1000)
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)
box(lty = 'solid', col = 'black')
abline(h = 0)"

R"p <- plot_diff(spr_model2, view = 'avg_surp', comp = list(factor=c(2,3)),
                 shade = TRUE, col.dif = '#FF0000', ylim = c(-60, 60), lwd = 3,
                 alpha = 0.4, ylab = '', main = '', hide.label = TRUE,
                 n.grid = 1000)
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)
box(lty = 'solid', col = 'black')
abline(h = 0)"

R"dev.off()"

###################### secondary experiment ####################################

spr_data_et = CSV.read("/home/danny/Documents/databases/next_word_prediction/MLM_results/results_spr_et_subtlex.csv",header = true)
spr_data_et2 = CSV.read("/home/danny/Documents/databases/next_word_prediction/MLM_results/results_spr_et2_subtlex.csv",header = true)

spr_data_et = prep_gam(spr_data_et, n_states, n_reps, method_names)
spr_data_et2 = prep_gam(spr_data_et2, n_states, n_reps, method_names)

spr_data_et_gam = spr_data_et[.!(spr_data_et[:fit_score] .< 0), :]
spr_data_et2_gam = spr_data_et2[.!(spr_data_et2[:fit_score] .< 0), :]

#conversion from julia categoricalarrays to R factors led to malformed factors
#so I perform this last preprocessing step in R
@rput(spr_data_et)
@rput(spr_data_et2)
@rput(spr_data_et_gam)
@rput(spr_data_et2_gam)

R"spr_data_et_gam$factor <- factor(spr_data_et_gam$factor, ordered = FALSE)"
R"spr_data_et_gam$rep <- factor(spr_data_et_gam$rep, ordered = FALSE)"

R"spr_data_et2_gam$factor <- factor(spr_data_et2_gam$factor, ordered = FALSE)"
R"spr_data_et2_gam$rep <- factor(spr_data_et2_gam$rep, ordered = FALSE)"

R"spr_data_et_model <- gam(fit_score ~ factor + s(avg_surp, by = factor, k = 30) +
                           s(avg_surp, rep, bs = 'fs', k = 2),
                           data = spr_data_et_gam[spr_data_et_gam$method == 'gru' |
                                                  spr_data_et_gam$method == 'gru2' |
                                                  spr_data_et_gam$method == 'tf' |
                                                  spr_data_et_gam$method == 'tf2',],
                           method = 'ML')"
R"gam.check(spr_data_et_model)"

R"spr_data_et2_model <- gam(fit_score ~ factor + s(avg_surp, by = factor, k = 30) +
                            s(avg_surp, rep, bs = 'fs', k = 3),
                            data = spr_data_et2_gam[spr_data_et2_gam$method == 'gru' |
                                                    spr_data_et2_gam$method == 'gru2' |
                                                    spr_data_et2_gam$method == 'tf' |
                                                    spr_data_et2_gam$method == 'tf2',],
                            method = 'ML')"
R"gam.check(spr_data_et2_model)"

R"jpeg('/home/danny/Pictures/gam_ET-EEG.jpg', res = 150, width = 1400,
       height = 1000)"

R"par(mfrow=c(2,2), mai = c(0.4, 0.4, 0.2, 0.1), cex = '1')"
R"plot(spr_data_et[spr_data_et$method == 'gru' | spr_data_et$method == 'tf' |
                   spr_data_et$method == 'gru2' | spr_data_et$method == 'tf2',]$avg_surp,
       spr_data_et[spr_data_et$method == 'gru' | spr_data_et$method == 'tf' |
                   spr_data_et$method == 'gru2' | spr_data_et$method == 'tf2',]$fit_score,
       main='SPR - short sentences', col = c(rep('red', 80),rep('green', 80),
       rep('cyan', 80),rep('purple', 80) ), xlab='', xlim = c(-8, -4.5),
       ylab='goodness-of-fit', pch = c(rep(1, 80),rep(2, 80), rep(3, 80),
                                       rep(4, 80)
                                       )
       )
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)
legend('topleft', bg = 'white', legend = c('2L GRU', 'GRU', '2L Transformer',
                                           'Transformer'),
       col=c('red', 'green', 'cyan', 'purple'), pch = c(1,2,3,4)
       )"

R"plot(spr_data_et2[spr_data_et2$method == 'gru' | spr_data_et2$method == 'tf' |
                    spr_data_et2$method == 'gru2' | spr_data_et2$method == 'tf2',]$avg_surp,
       spr_data_et2[spr_data_et2$method == 'gru' | spr_data_et2$method == 'tf' |
                    spr_data_et2$method == 'gru2' | spr_data_et2$method == 'tf2',]$fit_score,
       main='SPR - long sentences', col = c(rep('red', 80),rep('green', 80),
       rep('cyan', 80),rep('purple', 80) ), xlab='', xlim = c(-8, -4.5),
       ylab='', pch = c(rep(1, 80),rep(2, 80), rep(3, 80),rep(4, 80))
       )
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)"

R"p <- plot_smooth(spr_data_et_model, view= 'avg_surp', plot_all = 'factor',
                   se = TRUE, shade = TRUE, xlab = '', ylab = 'goodness-of-fit',
                   xlim = c(-8, -4.5), ylim = c(20, 120), lty = c(1,2,3,6),
                   lwd = 2, legend_plot_all = FALSE, rug = FALSE, alpha = 0.4,
                   hide.label = T)
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)
box(lty = 'solid', col = 'black')
legend('topleft', bg = 'white', legend = c('2L GRU','GRU', '2L Transformer',
                                           'Transformer'),
       col=c('red', 'green', 'cyan', 'purple'), lty = c(1,2,3,6), lwd = 2)"

R"p <- plot_smooth(spr_data_et2_model, view = 'avg_surp', plot_all = 'factor',
                   se = TRUE, shade = TRUE, xlim = c(-8, -4.5),
                   xlab = 'average surprisal', ylab = '', ylim = c(-30, 130),
                   lty = c(1,2,3,6), lwd = 2, legend_plot_all = FALSE,
                   rug = FALSE, alpha = 0.4, hide.label = T)
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)
box(lty = 'solid', col = 'black')"

R"dev.off()"

R"jpeg('/home/danny/Pictures/diff_ET-EEG.jpg', res = 150, width = 1400, height = 500)"

R"par(mfrow=c(1,2), mai = c(0.4, 0.4, 0.2, 0.1), cex = '1')"
R"p <- plot_diff(spr_data_et_model, view = 'avg_surp', comp = list(factor=c(1,2)),
                 col = '#FF0000', shade = TRUE, col.dif = '#FF0000',
                 ylim = c(-15, 15), lwd = 3, alpha = 0.4, xlim = c(-8, -4.5),
                 ylab = 'Estimated difference in goodness-of-fit score',
                 main = '', hide.label = TRUE, n.grid = 1000)
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)
box(lty = 'solid', col = 'black')"
R"p <- plot_diff(spr_data_et_model, view = 'avg_surp', comp = list(factor=c(6,4)),
                 add = TRUE, col = '#00FF00', shade = TRUE, xlim = c(-8, -4.5),
                 col.dif = '#00FF00', lty = 2, lwd = 3, alpha = 0.4,
                 n.grid = 1000)"
R"p <- plot_diff(spr_data_et_model, view = 'avg_surp', comp = list(factor=c(2,4)),
                 add = TRUE, col = '#0000FF', shade = TRUE, col.dif = '#0000FF',
                 lty = 3, lwd = 3, alpha = 0.4, n.grid = 1000, xlim = c(-8, -4.5))
abline(h = 0)
"

R"p <- plot_diff(spr_data_et2_model, view = 'avg_surp', comp = list(factor=c(1,2)),
                 col = '#FF0000', shade = TRUE, col.dif = '#FF0000',
                 ylim = c(-30, 40), lwd = 3, alpha = 0.4, xlim = c(-8, -4.5),
                 xlab = 'average surprisal', ylab = '', main = '',
                 hide.label = TRUE, n.grid = 1000)
grid(nx = NULL, ny = NULL, col = 'gray', lty = 'solid', lwd = par('lwd'),
     equilogs = TRUE)
box(lty = 'solid', col = 'black')"
R"p <- plot_diff(spr_data_et2_model, view = 'avg_surp', comp = list(factor=c(6,4)),
                 add = TRUE, col = '#00FF00', shade = TRUE, xlim = c(-8, -4.5),
                 col.dif = '#00FF00', lty = 2, lwd = 3, alpha = 0.4,
                 n.grid = 1000)"
R"p <- plot_diff(spr_data_et2_model, view = 'avg_surp', comp = list(factor=c(2,4)),
                 add = TRUE, col = '#0000FF', shade = TRUE, col.dif = '#0000FF',
                 lty = 3, lwd = 3, alpha = 0.4, n.grid = 1000, xlim = c(-8, -4.5))
abline(h = 0)
legend('top', bg = 'white',
       legend = c('2L GRU - GRU','Transformer - 2L Transformer', 'GRU - 2L Transformer'),
       col=c('#FF0000', '#00FF00', '#0000FF'), lty = c(1,2,3), lwd = 2, cex = 0.85)
"
R"dev.off()"
