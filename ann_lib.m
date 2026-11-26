function L = ann_lib()
 
    L.options = @options;
    L.load_concrete = @load_concrete;
    L.load_melbourne = @load_melbourne;
    L.network = @network;
    L.init_params = @init_params;
    L.predict = @predict;
    L.loss_grad = @loss_grad;
    L.train_gd = @train_gd;
    L.train_bfgs = @train_bfgs;
    L.train_adam = @train_adam;
    L.gradcheck = @gradcheck;
    L.tune = @tune;
    L.fit_final = @fit_final;
    L.evaluate = @evaluate;
    L.metrics = @metrics;
    L.acf = @sample_acf;
    L.quantile = @sample_quantile;
    L.explore = @explore;
    L.plot_results = @plot_results;
    L.write_results = @write_results;
    L.partition = @partition;
    L.start_history = @start_history;
    L.train_mini = @train_mini;
    L.algorithms = @algorithms;
    L.linear_fit = @linear_fit;
    L.new_fig = @new_fig;
    L.save_fig = @save_fig;
    L.hist_simple = @hist_simple;
    L.palette = @palette;
    L.progress_plot = @progress_plot;
    L.progress_overview = @progress_overview;
    L.draw_boxplots = @draw_boxplots;
    L.as_net = @as_net;
    L.set_seed = @set_seed;
    L.summary_text = @summary_text;
end

function o = options(quick)
    if nargin<1, quick = false; end
    o.seed = 42; o.budget = 500; o.batch = 32;          
    o.finalSeeds = 1:30; o.tuneSeeds = 101:105;
    o.rates.gd = logspace(-3,0,7);                       
    o.rates.adam = logspace(-4,-1,7);                    
    o.beta1 = 0.9; o.beta2 = 0.999; o.epsilon = 1e-8;
    o.armijo = 1e-4; o.shrink = 0.5; o.maxTrials = 20; o.tol = 1e-6;
    o.algorithms = {'GD','BFGS','Adam'};
    if quick, o.budget = 20; o.finalSeeds = 1:2; o.tuneSeeds = 101:102; end
end

function set_seed(s)
    if exist('rng','file') || exist('rng','builtin'), rng(s,'twister');
    else, rand('twister',s); randn('twister',s); end
end

function net = as_net(layers)
     
    if isstruct(layers), net = layers; return; end
    layers = layers(:)'; net.in = layers(1:end-1); net.out = layers(2:end);
    net.depth = numel(net.out); len = (net.in+1).*net.out;
    net.last = cumsum(len); net.first = [1,1+net.last(1:end-1)]; net.count = sum(len);
end