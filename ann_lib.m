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

function d = load_concrete(folder,o)
    a = dlmread(fullfile(folder,'concrete.csv'),',',1,0);
    assert(isequal(size(a),[1030 9]) && all(isfinite(a(:))), 'Unexpected Concrete data');
    [~,first] = unique(a,'rows','first');
    first = sort(first); clean = a(first,:);
    n = size(clean,1); set_seed(o.seed); order = randperm(n);
    ntr = floor(.70*n); nva = floor(.15*n);
    tr = order(1:ntr); va = order(ntr+(1:nva)); te = order(ntr+nva+1:end);
    x = clean(:,1:8); x(:,8) = log1p(x(:,8));
    d = partition(x,clean(:,9),tr,va,te);
    d.name = 'Concrete'; d.unit = 'MPa'; d.raw = a; d.cleaned = clean;
    d.sourceRows = first; d.testKnownMix = ismember(clean(te,1:7),clean(tr,1:7),'rows');
    d.featureNames = {'Cement','Slag','Fly ash','Water','Superplasticizer','Coarse aggregate','Fine aggregate','log(1+age)'};
end

function d = load_melbourne(folder,~)
    f = fopen(fullfile(folder,'melbourne.csv'),'rt');
    assert(f>=0,'Could not open melbourne.csv'); cleanup = onCleanup(@() fclose(f));
    raw = textscan(f,'%s','Delimiter',char(10),'Whitespace','');
    lines = raw{1}; record = ~cellfun('isempty',regexp(lines,'^\s*"?\d{4}-\d{2}-\d{2}"?\s*,','once'));
    fields = regexp(lines(record),'^\s*"?(\d{4}-\d{2}-\d{2})"?\s*,\s*"?([?]?[-+]?[0-9]+(?:\.[0-9]+)?)"?\s*$','tokens','once');
    assert(~any(cellfun('isempty',fields)),'Malformed numerical temperature record');
    dateText = cellfun(@(row) row{1},fields,'UniformOutput',false);
    tempText = cellfun(@(row) row{2},fields,'UniformOutput',false);
    dates = datenum(char(dateText),'yyyy-mm-dd');
    marked = ~cellfun('isempty',strfind(tempText,'?'));
    numericText = strrep(tempText,'?',''); 
    numericFields = textscan(sprintf('%s\n',numericText{:}),'%f'); values = numericFields{1};
    assert(numel(values)==3650 && all(isfinite(values)) && all(diff(dates)>0),'Unexpected Melbourne data');
    gridDates = (dates(1):dates(end))'; series = nan(size(gridDates));
    series(dates-dates(1)+1) = values;
    [yr,~,~] = datevec(gridDates);
    firstDay = datenum(yr,ones(size(yr)),ones(size(yr)));
    nextYear = datenum(yr+1,ones(size(yr)),ones(size(yr)));
    phase = 2*pi*(gridDates-firstDay)./(nextYear-firstDay);
    calendar = [sin(phase),cos(phase)];
    ntr = floor(.70*numel(dates)); nva = floor(.15*numel(dates));
    trEnd = dates(ntr); vaEnd = dates(ntr+nva); innerEnd = dates(floor(.8*ntr));
    candidates = [1 3 7 14]; maxLag = max(candidates); lags = nan(numel(series),maxLag);
    for j = 1:maxLag
        lags(j+1:end,j) = series(1:end-j); 
    end
    
    common = isfinite(series) & all(isfinite(lags),2);
    innerTrain = common & gridDates<=innerEnd;
    innerVal = common & gridDates>innerEnd & gridDates<=trEnd;
    scores = zeros(size(candidates));
    for i = 1:numel(candidates)
        design = [ones(numel(series),1),lags(:,1:candidates(i)),calendar];
        w = design(innerTrain,:) \ series(innerTrain);
        scores(i) = sqrt(mean((design(innerVal,:)*w-series(innerVal)).^2));
    end
    [~,i] = min(scores); p = candidates(i);
    x = [lags(:,1:p),calendar]; keep = isfinite(series) & all(isfinite(x),2);
    targetDates = gridDates(keep);
    tr = find(targetDates<=trEnd); va = find(targetDates>trEnd & targetDates<=vaEnd);
    te = find(targetDates>vaEnd); d = partition(x(keep,:),series(keep),tr,va,te);
    d.name = 'Melbourne'; d.unit = '°C'; d.dates = targetDates;
    d.gridDates = gridDates; d.series = series; d.obsDates = dates; d.values = values;
    d.flagDates = dates(marked);
    d.cutoffs = [innerEnd trEnd vaEnd]; d.p = p; d.candidates = candidates; d.lagScores = scores;
    d.featureNames = [arrayfun(@(k) sprintf('lag%d',k),1:p,'UniformOutput',false),{'sin','cos'}];
end

function d = partition(x,y,tr,va,te)
    d.tr = tr(:); d.va = va(:); d.te = te(:); d.x = x; d.y = y(:);
    d.mx = mean(x(tr,:),1); d.sx = std(x(tr,:),0,1); d.sx(d.sx==0) = 1;
    d.my = mean(y(tr)); d.sy = std(y(tr)); assert(d.sy>0);
    z = (x-d.mx)./d.sx; v = (y(:)-d.my)/d.sy;
    d.Xtr = z(tr,:)'; d.Xva = z(va,:)'; d.Xte = z(te,:)';
    d.Ytr = v(tr)'; d.Yva = v(va)'; d.ytest = y(te); d.dim = size(x,2);
end

function net = network(d,a)
    hidden = {10,100,10*ones(1,10)};
    width = [d,hidden{a},1]; net.in = width(1:end-1); net.out = width(2:end);
    net.depth = numel(net.out); len = (net.in+1).*net.out;
    net.last = cumsum(len); net.first = [1,1+net.last(1:end-1)];
    net.count = sum(len);
end

function theta = init_params(layers,seed)
    net = as_net(layers); set_seed(seed); theta = zeros(net.count,1);
    for k = 1:net.depth
        u = net.in(k); v = net.out(k);
        weights = (2*rand(v,u)-1)*sqrt(6/(u+v));
        block = [weights,zeros(v,1)]; theta(net.first(k):net.last(k)) = block(:);
    end
end

function [y,A,W] = predict(theta,net,x)
    A = cell(1,net.depth+1); W = cell(1,net.depth); A{1} = x;
    for k = 1:net.depth
        block = reshape(theta(net.first(k):net.last(k)),net.out(k),net.in(k)+1);
        W{k} = block(:,1:end-1);
        z = W{k}*A{k} + block(:,end);
        if k<net.depth, A{k+1} = tanh(z); else, A{k+1} = z; end
    end
    y = A{end};
end

function [f,g] = loss_grad(theta,net,x,y)
    [pred,A,W] = predict(theta,net,x); err = pred-y; f = mean(err.^2);
    if nargout<2, return; end
    delta = 2*err/numel(err); g = zeros(size(theta));
    for k = net.depth:-1:1
        block = [delta*A{k}',sum(delta,2)];
        g(net.first(k):net.last(k)) = block(:);
        if k>1, delta = (W{k}'*delta).*(1-A{k}.^2); end
    end
end

function [absMax,relMax,normRel] = gradcheck(net,d,seed)
    theta = init_params(net,seed); x = d.Xtr(:,1:8); y = d.Ytr(:,1:8);
    [~,g] = loss_grad(theta,net,x,y); num = zeros(size(g));
    for i = 1:numel(theta)
        h = 1e-5*max(1,abs(theta(i))); plus = theta; minus = theta;
        plus(i) = plus(i)+h; minus(i) = minus(i)-h;
        num(i) = (loss_grad(plus,net,x,y)-loss_grad(minus,net,x,y))/(2*h);
    end
    absMax = max(abs(g-num));
    relMax = max(abs(g-num)./max(1e-7,abs(g)+abs(num)));
    normRel = norm(g-num)/max(eps,norm(g)+norm(num));
end

function h = start_history(theta,net,Xtr,ytr,Xva,yva,o)
    h.train = nan(o.budget+1,1); h.val = h.train; h.seconds = h.train;
    h.train(1) = loss_grad(theta,net,Xtr,ytr);
    h.val(1) = loss_grad(theta,net,Xva,yva); h.seconds(1) = 0;
    h.bestVal = h.val(1); h.bestCost = 0; h.cost = 0; h.fullGrads = 0;
    h.batchGrads = 0; h.accepted = 0; h.failed = false; h.stop = 'budget';
end

function [best,h] = train_gd(theta,layers,Xtr,ytr,Xva,yva,o)
    [best,h] = train_mini(theta,layers,Xtr,ytr,Xva,yva,o,false);
end

function [best,h] = train_adam(theta,layers,Xtr,ytr,Xva,yva,o)
    [best,h] = train_mini(theta,layers,Xtr,ytr,Xva,yva,o,true);
end

function [best,h] = train_mini(theta,layers,Xtr,ytr,Xva,yva,o,adam)
    net = as_net(layers);
    if ~isfield(o,'orderSeed'), o.orderSeed = o.seed; end
    h = start_history(theta,net,Xtr,ytr,Xva,yva,o); best = theta; clock = tic;
    set_seed(o.orderSeed); n = size(Xtr,2);
    m = zeros(size(theta)); v = m;
    for ep = 1:o.budget
        order = randperm(n);
        for first = 1:o.batch:n
            ix = order(first:min(n,first+o.batch-1));
            [~,g] = loss_grad(theta,net,Xtr(:,ix),ytr(ix));
            h.batchGrads = h.batchGrads+1;
            if adam
                m = o.beta1*m+(1-o.beta1)*g; v = o.beta2*v+(1-o.beta2)*(g.^2);
                mh = m/(1-o.beta1^h.batchGrads); vh = v/(1-o.beta2^h.batchGrads);
                theta = theta-o.rate*mh./(sqrt(vh)+o.epsilon);
            else
                theta = theta-o.rate*g;
            end
        end
        h.cost = ep; h.train(ep+1) = loss_grad(theta,net,Xtr,ytr);
        h.val(ep+1) = loss_grad(theta,net,Xva,yva);
        if ~isfinite(h.train(ep+1)) || ~isfinite(h.val(ep+1))
            h.failed = true; h.stop = 'nonfinite'; h.seconds(ep+1) = toc(clock); break;
        end
        if h.val(ep+1)<h.bestVal, h.bestVal = h.val(ep+1); h.bestCost = ep; best = theta; end
        h.accepted = ep; h.seconds(ep+1) = toc(clock);
    end
    h.elapsed = toc(clock);
    h.fullGrads = h.batchGrads*o.batch/n;
end

function [best,h] = train_bfgs(theta,layers,Xtr,ytr,Xva,yva,o)
    net = as_net(layers);
    h = start_history(theta,net,Xtr,ytr,Xva,yva,o); best = theta; clock = tic;
    P = net.count; H = eye(P); first = true; [f,g] = loss_grad(theta,net,Xtr,ytr);
    h.cost = 1; h.fullGrads = 1; h.train(2) = f; h.val(2) = h.val(1); h.seconds(2) = toc(clock);
    while h.cost<o.budget
        if norm(g,inf)<=o.tol, h.stop = 'tolerance'; break; end
        direction = -(H*g); slope = g'*direction;
        if ~isfinite(slope) || slope>=0, H = eye(P); first = true; direction = -g; slope = -(g'*g); end
        if first, alpha = min(1,1/norm(g)); else, alpha = 1; end
        accepted = false;
        for trial = 1:o.maxTrials
            candidate = theta+alpha*direction;
            [fc,gc] = loss_grad(candidate,net,Xtr,ytr);
            h.cost = h.cost+1; h.fullGrads = h.fullGrads+1; i = h.cost+1;
            h.train(i) = h.train(i-1); h.val(i) = h.val(i-1);
            if isfinite(fc) && all(isfinite(gc)) && fc<=f+o.armijo*alpha*slope
                s = candidate-theta; y = gc-g; ys = y'*s;
                if ys>1e-10*norm(y)*norm(s)
                    if first, H = (ys/(y'*y))*eye(P); first = false; end
                    Hy = H*y;
                    v = ((1+(y'*Hy)/ys)/(2*ys))*s-Hy/ys;
                    H = H + s*v' + v*s';
                end
                theta = candidate; g = gc; f = fc; h.train(i) = f;
                h.val(i) = loss_grad(theta,net,Xva,yva); h.accepted = h.accepted+1;
                if h.val(i)<h.bestVal, h.bestVal = h.val(i); h.bestCost = h.cost; best = theta; end
                accepted = true;
            end
            h.seconds(i) = toc(clock);
            if accepted || h.cost>=o.budget, break; end
            alpha = alpha*o.shrink;
        end
        if ~accepted && h.cost<o.budget, h.stop = 'line_search'; h.failed = true; break; end
    end
    h.elapsed = toc(clock);
end

function fit = algorithms()
    fit = {@train_gd,@train_bfgs,@train_adam};
end

function t = tune(d,net,o)
    fit = algorithms(); t.rates = [o.rates.gd(:),o.rates.adam(:)];
    t.scores = nan(size(t.rates,1),numel(o.tuneSeeds),2); t.selected = nan(1,3);
    for j = 1:2
        alg = 2*j-1;
        for r = 1:numel(o.tuneSeeds)
            seed = o.tuneSeeds(r); theta0 = init_params(net,seed); o.orderSeed = o.seed+1000*seed;
            for v = 1:size(t.rates,1)
                o.rate = t.rates(v,j);
                [~,h] = fit{alg}(theta0,net,d.Xtr,d.Ytr,d.Xva,d.Yva,o);
                if h.failed, t.scores(v,r,j) = Inf; else, t.scores(v,r,j) = h.bestVal; end
            end
        end
        [~,k] = min(mean(t.scores(:,:,j),2)); t.selected(alg) = t.rates(k,j);
        if any(t.selected(alg)==t.rates([1 end],j))
            fprintf('  AVISO: eta de %s no limite da grelha (%g)\n',o.algorithms{alg},t.selected(alg));
        end
    end
end

function block = fit_final(d,net,t,o)
    fit = algorithms(); ns = numel(o.finalSeeds);
    block.theta = cell(3,ns); block.history = cell(3,ns); block.initialLayerNorm = nan(ns,2);
    for i = 1:ns
        seed = o.finalSeeds(i); theta0 = init_params(net,seed); o.orderSeed = o.seed+1000*seed;
        [~,g] = loss_grad(theta0,net,d.Xtr,d.Ytr);
        block.initialLayerNorm(i,:) = [norm(g(net.first(1):net.last(1))),norm(g(net.first(end):net.last(end)))];
        for alg = 1:3
            o.rate = t.selected(alg);
            [th,h] = fit{alg}(theta0,net,d.Xtr,d.Ytr,d.Xva,d.Yva,o);
            if ~strcmp(h.stop,'nonfinite')
                h.train(h.cost+2:end) = h.train(h.cost+1);
                h.val(h.cost+2:end)   = h.val(h.cost+1);
            end
            block.theta{alg,i} = th; block.history{alg,i} = h;
        end
    end
end

function m = metrics(y,p)
    y = y(:); p = p(:); e = p-y; mse = mean(e.^2);
    m = [mse,sqrt(mse),mean(abs(e)),1-sum(e.^2)/sum((y-mean(y)).^2)];
end

function e = evaluate(d,blocks,o)
    ns = numel(o.finalSeeds); vmean = zeros(3,3); e.bestVals = cell(3,3);
    for a = 1:3
        for j = 1:3
            vv = cellfun(@(h) h.bestVal,blocks{a}.history(j,:));
            e.bestVals{a,j} = vv; vmean(a,j) = mean(vv);
        end
    end
    [~,winner] = min(vmean(:)); [wa,wj] = ind2sub([3,3],winner);
    v = e.bestVals{wa,wj}; [~,rep] = min(abs(v-median(v)));
    e.choice = [wa,wj,rep]; e.scores = zeros(3,3,ns,4);
    for a = 1:3
        net = network(d.dim,a);
        for j = 1:3
            e.pred{a,j} = zeros(numel(d.te),ns);
            for r = 1:ns
                p = predict(blocks{a}.theta{j,r},net,d.Xte)'*d.sy+d.my;
                e.pred{a,j}(:,r) = p;
                e.scores(a,j,r,:) = metrics(d.ytest,p);
            end
        end
    end
    if strcmp(d.name,'Concrete')
        e.refNames = {'OLS_raw_age','OLS_log_age'};
        e.refPred = [linear_fit(d.cleaned(:,1:8),d),linear_fit(d.x,d)];
    else
        e.refNames = {'Persistence','AR_lags','AR_lags_calendar'};
        e.refPred = [d.x(d.te,1),linear_fit(d.x(:,1:d.p),d),linear_fit(d.x,d)];
    end
    for j = 1:numel(e.refNames)
        e.refScores(j,:) = metrics(d.ytest,e.refPred(:,j));
    end
end

function p = linear_fit(x,d)
    design = [ones(size(x,1),1),x];
    coef = design(d.tr,:) \ d.y(d.tr);
    p = design(d.te,:)*coef;
end