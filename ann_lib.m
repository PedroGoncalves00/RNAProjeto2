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

function q = sample_quantile(v,p)
    v = sort(v(isfinite(v))); q = nan(size(p));
    if isempty(v), return; end
    x = 1+(numel(v)-1)*p;
    for i = 1:numel(p)
        low = floor(x(i)); high = ceil(x(i));
        q(i) = v(low)+(x(i)-low)*(v(high)-v(low));
    end
end

function c = sample_acf(x,maxLag)
    % Pearson correlation of available endpoints exactly j calendar days apart.
    c = nan(maxLag+1,1);
    for j = 0:maxLag
        a = x(1:end-j); b = x(1+j:end); ok = isfinite(a)&isfinite(b);
        if sum(ok)<2, continue; end
        a = a(ok)-mean(a(ok)); b = b(ok)-mean(b(ok));
        c(j+1) = sum(a.*b)/sqrt(sum(a.^2)*sum(b.^2));
    end
end

function f = new_fig(w,h)
    f = figure('Visible','off','Color','w','Units','pixels','Position',[60 60 w h]);
    set(f,'DefaultAxesFontName','DejaVu Sans','DefaultTextFontName','DejaVu Sans',...
          'DefaultAxesFontSize',10,'DefaultAxesBox','off','PaperPositionMode','auto');
end

function save_fig(f,folder,name)
    if ~exist(folder,'dir'), mkdir(folder); end
    print(f,fullfile(folder,[name '.png']),'-dpng','-r140'); close(f);
end

function hist_simple(x,n,color)
    [counts,centres] = hist(x(isfinite(x)),n);
    bar(centres,counts,1,'FaceColor',color,'EdgeColor','none'); axis tight;
end

function explore(d,folder)
    teal = [0.03 .43 .49]; blue = [.12 .35 .61]; rust = [.76 .31 .13];
    if strcmp(d.name,'Concrete')
        names = {'Cimento','Escória','Cinza volante','Água','Superplast.','Agreg. grosso','Agreg. fino','Idade','Resistência'};
        units = [repmat({'kg/m^3'},1,7),{'dias','MPa'}];
        f = new_fig(845,620);
        for i = 1:9
            subplot(3,3,i); hist_simple(d.raw(:,i),18,teal);
            title(names{i}); xlabel(units{i}); ylabel('Contagem');
        end
        save_fig(f,folder,'concrete_hist');
        f = new_fig(840,460);
        ticks = {[100 300 500],[0 150 300],[0 100 200],[130 180 230],[0 15 30],[800 950 1100],[600 800 1000],[1 180 365]};
        for i = 1:8
            subplot(2,4,i); plot(d.raw(:,i),d.raw(:,9),'.','Color',blue,'MarkerSize',3);
            xlabel([names{i} ' (' units{i} ')']); ylabel('MPa'); axis tight; set(gca,'XTick',ticks{i});
        end
        save_fig(f,folder,'concrete_scatter');
        f = new_fig(500,420); R = corrcoef(d.raw); imagesc(R,[-1 1]); axis image;
        k = 65; cold = [.13,.32,.62]; hot = [.74,.25,.17];
        map = [linspace(cold(1),1,k)',linspace(cold(2),1,k)',linspace(cold(3),1,k)';...
               linspace(1,hot(1),k)',linspace(1,hot(2),k)',linspace(1,hot(3),k)']; colormap(map); colorbar;
        short = {'Cim','Esc','Cin','Agu','Sup','Gro','Fin','Ida','Res'};
        set(gca,'XTick',1:9,'XTickLabel',short,'YTick',1:9,'YTickLabel',short);
        for i=1:9
            for j=1:9
                col=[.13 .14 .16]; if abs(R(i,j))>.68, col=[1 1 1]; end
                text(j,i,sprintf('%.2f',R(i,j)),'HorizontalAlignment','center','Color',col,'FontSize',8);
            end
        end
        save_fig(f,folder,'concrete_corr'); return;
    end
    times = d.gridDates; val = d.series; obs = d.obsDates; y = d.values;
    [~,mo,dy] = datevec(obs); months = accumarray(mo,y,[12 1],@mean);
    anchor = datenum(2000*ones(size(mo)),mo,dy)-datenum(2000,1,1)+1;
    cycle = accumarray(anchor,y,[366 1],@mean,NaN); smooth = nan(366,1);
    for i=1:366
        v = cycle(mod((i-15:i+15)-1,366)+1); smooth(i) = mean(v(isfinite(v)));
    end
    dif = diff(val); finiteDiff = dif(isfinite(dif)); center = median(finiteDiff);
    thresh = 3*1.4826*median(abs(finiteDiff-center)); flags = abs(dif-center)>thresh;
    f = new_fig(910,590);
    subplot(2,2,1); plot(times,val,'Color',teal,'LineWidth',.55); hold on;
    plot(times([false;flags]),val([false;flags]),'.','Color',rust,'MarkerSize',5);
    set(gca,'XTick',datenum(1981:2:1991,1,1)); datetick('x','yyyy','keepticks'); xlim([times(1) times(end)]);
    ylabel('Temperatura (°C)'); title('Série diária e saltos sinalizados');
    subplot(2,2,2); hist_simple(y,24,teal); xlabel('Temperatura (°C)'); ylabel('Dias'); title('Distribuição');
    subplot(2,2,3); bar(1:12,months,.8,'FaceColor',teal,'EdgeColor','none');
    set(gca,'XTick',1:12,'XTickLabel',{'J','F','M','A','M','J','J','A','S','O','N','D'}); xlim([.3 12.7]);
    ylabel('Média (°C)'); title('Médias mensais');
    subplot(2,2,4); plot(1:366,cycle,'Color',[.65 .79 .80]); hold on;
    plot(1:366,smooth,'Color',blue,'LineWidth',1.9); xlim([1 366]); xlabel('Dia do calendário'); ylabel('Média (°C)');
    title('Ciclo anual e média móvel de 31 dias');
    save_fig(f,folder,'melbourne_eda');
    f = new_fig(900,302);
    subplot(1,2,1); c = sample_acf(val,400); plot(0:400,c,'Color',teal,'LineWidth',1.25); hold on;
    plot([0 400],[0 0],'k:'); xlim([0 400]); ylim([-.8 1.05]); xlabel('Desfasamento (dias)'); ylabel('Correlação'); title('Autocorrelação');
    subplot(1,2,2); plot(val(1:end-1),val(2:end),'.','Color',blue,'MarkerSize',3);
    xlabel('T(t) (°C)'); ylabel('T(t+1) (°C)'); title('Dias consecutivos'); axis tight;
    save_fig(f,folder,'melbourne_dependencia');
end

function colors = palette()
    colors = [.10 .36 .68; .78 .32 .10; .02 .47 .40];
end

function progress_plot(block,o,folder,name,byTime)
    f = new_fig(760,320); colors = palette(); fields = {'train','val'};
    for panel = 1:2
        subplot(1,2,panel); hold on; grids = cell(1,3); mid = grids;
        for j = 1:3
            runs = block.history(j,:); ns = numel(runs);
            if byTime
                gridx = linspace(0,median(cellfun(@(h) h.elapsed,runs)),65);
                values = nan(numel(gridx),ns);
                for r=1:ns
                    h=runs{r}; tt=h.seconds(1:h.cost+1); v=h.(fields{panel});
                    for ix=1:numel(gridx)
                        at = find(tt<=gridx(ix),1,'last'); values(ix,r)=v(at);
                    end
                end
            else
                gridx = 0:o.budget; values = nan(numel(gridx),ns);
                for r=1:ns
                    values(:,r) = runs{r}.(fields{panel});
                end
            end
            q = nan(numel(gridx),3);
            for ix=1:numel(gridx), q(ix,:)=sample_quantile(values(ix,:),[.25 .5 .75]); end
            mask=all(isfinite(q),2); gx=gridx(mask); qq=q(mask,:);
            shade = .77+.23*colors(j,:);
            fill([gx(:);flipud(gx(:))],[qq(:,1);flipud(qq(:,3))],shade,'EdgeColor','none','FaceAlpha',.52);
            grids{j} = gx; mid{j} = qq(:,2);
        end
        lines = cell(1,3);
        for j=1:3
            lines{j}=plot(grids{j},mid{j},'Color',colors(j,:),'LineWidth',2.35);
        end
        set(gca,'YScale','log','YLim',o.plotLimits,'YTick',o.plotTicks,...
            'YTickLabel',arrayfun(@(v) sprintf('%g',v),o.plotTicks,'UniformOutput',false),...
            'YGrid','on','XGrid','off','GridLineStyle',':','GridColor',[.72 .76 .79]);
        ylabel('MSE normalizado');
        if byTime, xlabel('Tempo (s)'); else, xlabel('Épocas equivalentes'); xlim([0 o.budget]); end
        if panel==1, title('Treino'); else, title('Validação'); end
        if panel==1, legend([lines{:}],o.algorithms,'Location','northeast','FontSize',8); end
    end
    save_fig(f,folder,name);
end

function progress_overview(d,blocks,o,folder)
    
    f=new_fig(1150,450); colors=palette(); names={'A  Referência','B  Larga','C  Profunda'};
    fields={'train','val'}; rows={'Treino (MSE)','Validação (MSE)'};
    ticks = o.plotTicks;
    xt = round(linspace(0,o.budget,5));
    for row=1:2
        for a=1:3
            left=.075+(a-1)*.32; bottom=.57-(row-1)*.43;
            axes('Position',[left bottom .263 .31]); hold on;
            med=cell(1,3); xx=0:o.budget;
            for j=1:3
                vs=cellfun(@(h) h.(fields{row}),blocks{a}.history(j,:),'UniformOutput',false);
                v=[vs{:}]; q=nan(size(v,1),3);
                for i=1:size(v,1), q(i,:)=sample_quantile(v(i,:),[.25 .5 .75]); end
                med{j}=q(:,2); ok=isfinite(q(:,2)); gx=xx(ok); q=q(ok,:);
                fill([gx(:);flipud(gx(:))],[q(:,1);flipud(q(:,3))],.73+.27*colors(j,:),'EdgeColor','none','FaceAlpha',.55);
            end
            for j=1:3, plot(xx,med{j},'LineWidth',2.7,'Color',colors(j,:)); end
            set(gca,'FontSize',14,'YScale','log','YLim',o.plotLimits,'YTick',ticks,...
                'YTickLabel',arrayfun(@(v) sprintf('%g',v),ticks,'UniformOutput',false),...
                'XTick',xt,'XLim',[0 o.budget],'YGrid','on','GridLineStyle',':','TickDir','out');
            if row==1, title(names{a},'FontSize',17); set(gca,'XTickLabel',{}); end
            if a==1, ylabel(rows{row},'FontSize',14); end
            if row==2 && a==2, xlabel('Épocas equivalentes','FontSize',14); end
        end
    end
    save_fig(f,folder,[lower(d.name) '_convergencia_conjunta']);
end

function draw_boxplots(matrix,labels)
    colors = palette(); hold on;
    for k=1:size(matrix,2)
        v = sort(matrix(:,k)); q=sample_quantile(v,[.25 .5 .75]); gap=q(3)-q(1);
        lo=min(v(v>=q(1)-1.5*gap)); hi=max(v(v<=q(3)+1.5*gap)); c=colors(1+mod(k-1,3),:);
        fill(k+[-.30 .30 .30 -.30],[q(1) q(1) q(3) q(3)],.7+.3*c,'EdgeColor',c);
        plot([k-.30 k+.30],[q(2) q(2)],'Color',c,'LineWidth',2);
        plot([k k],[lo q(1)],'Color',c); plot([k k],[q(3) hi],'Color',c);
        plot(k+[-.15 .15],[lo lo],'Color',c); plot(k+[-.15 .15],[hi hi],'Color',c);
        out=v(v<lo | v>hi); plot(k*ones(size(out)),out,'o','Color',c,'MarkerSize',3);
    end
    set(gca,'XTick',1:numel(labels),'XTickLabel',labels); xlim([.4 numel(labels)+.6]);
    set(gca,'YGrid','on','GridLineStyle',':','GridColor',[.72 .76 .79]);
end

function plot_results(d,blocks,e,o,folder)
    prefix=lower(d.name); ns=numel(o.finalSeeds); boxes=zeros(ns,9); labs=cell(1,9);
    
    bound=[Inf -Inf];
    for a=1:3
        for j=1:3
            for field={'train','val'}
                m=cellfun(@(h) h.(field{1}),blocks{a}.history(j,:),'UniformOutput',false);
                values=[m{:}];
                for i=1:size(values,1)
                    q=sample_quantile(values(i,:),[.25 .75]);
                    if all(isfinite(q)), bound=[min(bound(1),q(1)) max(bound(2),q(2))]; end
                end
            end
        end
    end
    o.plotLimits=[.9*bound(1) 1.15*bound(2)];
    ticks=[.02 .04 .08 .1 .2 .3 .4 .6 .8 1 1.5 2 3 5 10];
    o.plotTicks=ticks(ticks>o.plotLimits(1) & ticks<o.plotLimits(2));
    if numel(o.plotTicks)<2
        o.plotTicks = logspace(log10(o.plotLimits(1)),log10(o.plotLimits(2)),4);
    end
    for a=1:3
        suffix = char(64+a);
        progress_plot(blocks{a},o,folder,[prefix '_curva_' suffix],false);
        progress_plot(blocks{a},o,folder,[prefix '_tempo_' suffix],true);
        for j=1:3
            k=3*(a-1)+j; boxes(:,k)=reshape(e.scores(a,j,:,2),[],1);
            labs{k}=[suffix '-' o.algorithms{j}];
        end
    end
    progress_overview(d,blocks,o,folder);
    f=new_fig(900,360); draw_boxplots(boxes,labs); ylabel(['RMSE (' d.unit ')']); save_fig(f,folder,[prefix '_boxes']);
    win=e.choice; y=d.ytest(:); p=e.pred{win(1),win(2)}(:,win(3)); err=p-y;
    blue=[.10 .36 .68]; rust=[.78 .32 .10]; f=new_fig(860,340);
    if strcmp(d.name,'Concrete')
        subplot(1,2,1); plot(y,p,'o','Color',blue,'MarkerSize',3); hold on;
        span=[min([y;p]) max([y;p])]; plot(span,span,'k--'); xlim(span); ylim(span);
        xlabel('Medido (MPa)'); ylabel('Previsto (MPa)'); title('Resistência no teste'); axis square;
        subplot(1,2,2); hist_simple(err,16,blue); xlabel('Previsto - medido (MPa)'); ylabel('Amostras'); title('Resíduos');
    else
        close(f); f=new_fig(880,590);
        subplot(2,2,[1 2]); dates=d.dates(d.te); plot(dates,y,'Color',[.40 .45 .49]); hold on;
        plot(dates,p,'Color',rust); datetick('x','mmm yy'); xlim([dates(1) dates(end)]);
        ylabel('°C'); title('Previsão diária no teste'); legend({'Observada','Prevista'},'Location','northeast');
        subplot(2,2,3); nz=min(42,numel(y)); plot(dates(1:nz),y(1:nz),'Color',[.35 .4 .44]); hold on;
        plot(dates(1:nz),p(1:nz),'Color',rust,'LineWidth',1.45); datetick('x','dd/mm'); xlim([dates(1) dates(nz)]);
        xlabel('Dia/mes'); ylabel('°C'); title('Primeiros 42 dias');
        subplot(2,2,4); hist_simple(err,20,blue); xlabel('Prevista - observada (°C)'); ylabel('Dias'); title('Resíduos');
    end
    save_fig(f,folder,[prefix '_previsoes']);
end


function write_results(data,tunings,blocks,ev,checks,o,out)
    if ~exist(out,'dir'), mkdir(out); end
    names={'execucoes','historico','afinacao','resumo','referencias','previsoes','gradientes','particoes','estatisticas'};
    heads={...
     'dataset,arch,algorithm,seed,test_mse,test_rmse,test_mae,test_r2,elapsed_s,best_cost,best_val,cost,full_grad_calls,minibatch_grad_calls,accepted,failed,stopping', ...
     'dataset,arch,algorithm,seed,epoch_equiv,seconds,train_mse_scaled,val_mse_scaled', ...
     'dataset,arch,algorithm,learning_rate,tune_seed,best_val_scaled,selected', ...
     'dataset,arch,algorithm,rmse_mean,rmse_sd,rmse_median,rmse_best,rmse_worst,r2_mean,r2_sd,r2_median,r2_best,r2_worst,time_mean_s', ...
     'dataset,reference,test_mse,test_rmse,test_mae,test_r2', ...
     'dataset,source_record_or_date,arch,algorithm,seed,measured,predicted,residual_pred_minus_measured', ...
     'dataset,arch,params,max_absolute,max_relative,relative_norm', ...
     'dataset,partition,source_record_or_target_date', ...
     'dataset,variable,min,q1,median,q3,max,mean,sd'};
    f=zeros(1,numel(names)); guards=cell(size(f));
    for i=1:numel(names)
        f(i)=fopen(fullfile(out,[names{i} '.csv']),'wt'); assert(f(i)>=0,'Cannot write results');
        handle=f(i); guards{i}=onCleanup(@() fclose(handle)); fprintf(f(i),'%s\n',heads{i});
    end
    for k=1:numel(data)
        d=data{k}; e=ev{k}; tag=d.name;
        if k==1
            keys=arrayfun(@(i) sprintf('%d',i),d.sourceRows,'UniformOutput',false);
            vals=d.raw; feats={'cement','slag','fly_ash','water','superplasticizer','coarse_aggregate','fine_aggregate','age','strength'};
        else
            keys=cellstr(datestr(d.dates,'yyyy-mm-dd')); vals=d.values; feats={'temperature'};
        end
        for j=1:size(vals,2)
            v=vals(:,j); q=sample_quantile(v,[0 .25 .5 .75 1]);
            fprintf(f(9),'%s,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n',tag,feats{j},q,mean(v),std(v));
        end
        sets={d.tr,d.va,d.te}; labels={'train','validation','test'};
        for j=1:3
            for idx=sets{j}(:)', fprintf(f(8),'%s,%s,%s\n',tag,labels{j},keys{idx}); end
        end
        for a=1:3
            A=char(64+a); t=tunings{k,a}; block=blocks{k,a}; net=network(d.dim,a);
            fprintf(f(7),'%s,%s,%d,%.12g,%.12g,%.12g\n',tag,A,net.count,reshape(checks(k,a,:),1,3));
            for jj=1:2
                j=2*jj-1;
                for vi=1:size(t.rates,1)
                    for r=1:numel(o.tuneSeeds)
                        fprintf(f(3),'%s,%s,%s,%.12g,%d,%.12g,%d\n',tag,A,o.algorithms{j},t.rates(vi,jj),o.tuneSeeds(r),t.scores(vi,r,jj),t.rates(vi,jj)==t.selected(j));
                    end
                end
            end
            for j=1:3
                for r=1:numel(o.finalSeeds)
                    h=block.history{j,r}; m=reshape(e.scores(a,j,r,:),1,4);
                    fprintf(f(1),'%s,%s,%s,%d,%.12g,%.12g,%.12g,%.12g,%.9g,%d,%.12g,%d,%.6g,%d,%d,%d,%s\n',...
                      tag,A,o.algorithms{j},o.finalSeeds(r),m,h.elapsed,h.bestCost,h.bestVal,h.cost,h.fullGrads,h.batchGrads,h.accepted,h.failed,h.stop);
                    for c=0:h.cost
                        fprintf(f(2),'%s,%s,%s,%d,%d,%.9g,%.12g,%.12g\n',tag,A,o.algorithms{j},o.finalSeeds(r),c,h.seconds(c+1),h.train(c+1),h.val(c+1));
                    end
                end
                rm=reshape(e.scores(a,j,:,2),[],1); r2=reshape(e.scores(a,j,:,4),[],1);
                ts=cellfun(@(h) h.elapsed,block.history(j,:));
                fprintf(f(4),'%s,%s,%s,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g,%.9g\n',...
                     tag,A,o.algorithms{j},mean(rm),std(rm),median(rm),min(rm),max(rm),mean(r2),std(r2),median(r2),max(r2),min(r2),mean(ts));
            end
        end
        for j=1:numel(e.refNames)
            fprintf(f(5),'%s,%s,%.12g,%.12g,%.12g,%.12g\n',tag,e.refNames{j},e.refScores(j,:));
        end
        c=e.choice; p=e.pred{c(1),c(2)}(:,c(3));
        for i=1:numel(p)
            fprintf(f(6),'%s,%s,%s,%s,%d,%.12g,%.12g,%.12g\n',tag,keys{d.te(i)},char(64+c(1)),o.algorithms{c(2)},o.finalSeeds(c(3)),d.ytest(i),p(i),p(i)-d.ytest(i));
        end
    end
end

function summary_text(data,tunings,blocks,ev,checks,o,out)
    ln = {}; nm = 'ABC'; ns = numel(o.finalSeeds);
    for k = 1:numel(data)
        d = data{k}; e = ev{k}; B = blocks(k,:);
        ln{end+1} = sprintf('==== %s ====',d.name);
        if strcmp(d.name,'Concrete')
            ln{end+1} = sprintf('Linhas brutas: %d | duplicados removidos: %d | usadas: %d', ...
                size(d.raw,1),size(d.raw,1)-size(d.cleaned,1),size(d.cleaned,1));
            ln{end+1} = sprintf('Teste cuja mistura (7 colunas) tambem esta no treino: %.1f%%',100*mean(d.testKnownMix));
        else
            ln{end+1} = sprintf('Janela p=%d | candidatos %s | RMSE interno %s',d.p,mat2str(d.candidates),mat2str(d.lagScores,4));
            ln{end+1} = sprintf('Valores marcados com ? no ficheiro: %d',numel(d.flagDates));
        end
        for a = 1:3
            net = network(d.dim,a); t = tunings{k,a};
            ln{end+1} = sprintf('Arq %s (%d params): gradcheck rel.max=%.2e norm.rel=%.2e | eta GD=%g | eta Adam=%g', ...
                nm(a),net.count,checks(k,a,2),checks(k,a,3),t.selected(1),t.selected(3));
        end
        for m = [2 4]
            if m==2, ln{end+1} = sprintf('RMSE de teste (%s), media +- dp',d.unit);
            else, ln{end+1} = 'R2 de teste, media +- dp'; end
            ln{end+1} = sprintf('%-6s %-22s %-22s %-22s','Arq',o.algorithms{:});
            for a = 1:3
                c = cell(1,3);
                for j = 1:3
                    v = reshape(e.scores(a,j,:,m),[],1); c{j} = sprintf('%.4f +- %.4f',mean(v),std(v));
                end
                ln{end+1} = sprintf('%-6s %-22s %-22s %-22s',nm(a),c{:});
            end
        end
        for j = 1:numel(e.refNames)
            ln{end+1} = sprintf('Referencia %-18s RMSE=%.4f MAE=%.4f R2=%.4f', ...
                e.refNames{j},e.refScores(j,2),e.refScores(j,3),e.refScores(j,4));
        end
        ln{end+1} = 'Diagnosticos (falhas inclui paragem da pesquisa em linha; gap = MSE val - MSE treino no melhor ponto, normalizado):';
        for a = 1:3
            gr = median(B{a}.initialLayerNorm(:,1)./B{a}.initialLayerNorm(:,2));
            for j = 1:3
                H = B{a}.history(j,:);
                nfail = sum(cellfun(@(h) h.failed,H));
                bc  = median(cellfun(@(h) h.bestCost,H));
                gap = median(cellfun(@(h) h.bestVal-h.train(h.bestCost+1),H));
                spe = mean(cellfun(@(h) h.elapsed/max(h.cost,1),H));
                ln{end+1} = sprintf('%s-%-4s falhas=%d/%d melhor_epoca=%g gap=%.4f s/epoca=%.4f |g1|/|gL|=%.3g', ...
                    nm(a),o.algorithms{j},nfail,ns,bc,gap,spe,gr);
            end
        end
        ln{end+1} = '';
    end
    f = fopen(fullfile(out,'resumo.txt'),'wt'); assert(f>=0,'Cannot write resumo.txt');
    fprintf(f,'%s\n',ln{:}); fclose(f); fprintf('%s\n',ln{:});
end