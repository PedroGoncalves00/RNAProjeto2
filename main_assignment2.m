projectDir = fileparts(mfilename('fullpath'));
if isempty(projectDir), projectDir = pwd; end
addpath(projectDir);
if exist('OCTAVE_VERSION','builtin')
    more off;
    tk = available_graphics_toolkits();
    if any(strcmp(tk,'qt')), graphics_toolkit('qt');
    elseif any(strcmp(tk,'fltk')), graphics_toolkit('fltk'); end
end
L = ann_lib();
quick = exist('QUICK','var') && QUICK;
o = L.options(quick); L.set_seed(o.seed);
inputDir = fullfile(projectDir,'dados'); outputDir = fullfile(projectDir,'resultados');
figureDir = fullfile(outputDir,'figuras');
if ~exist(outputDir,'dir'), mkdir(outputDir); end
if ~exist(figureDir,'dir'), mkdir(figureDir); end
data = {L.load_concrete(inputDir,o),L.load_melbourne(inputDir,o)};
tunings = cell(2,3); blocks = cell(2,3); checks = zeros(2,3,3);

for k=1:2
    d=data{k}; fprintf('%s: train=%d, validation=%d, test=%d\n',d.name,numel(d.tr),numel(d.va),numel(d.te));
    L.explore(d,figureDir);
    for a=1:3
        net=L.network(d.dim,a);
        [ea,er,en]=L.gradcheck(net,d,o.seed); checks(k,a,:)=[ea,er,en];
        assert(en<1e-6,'Backpropagation failed the finite-difference check');
        fprintf('Tuning %s / %s (%d weights and biases)\n',d.name,char(64+a),net.count);
        tunings{k,a}=L.tune(d,net,o);
    end
end