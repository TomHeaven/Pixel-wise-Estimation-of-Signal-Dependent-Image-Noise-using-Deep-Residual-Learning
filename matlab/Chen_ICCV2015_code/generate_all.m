datasets = {'kodak', 'mcm' 'bsd500'};
%datasets = {'kodak', 'mcm'};
%sigmas = [5, 15,  25];
sigmas = [15];

lenData = length(datasets);
lenNoise = length(sigmas);

for i = 1:lenData
    dataset = datasets{i};
    for j = 1:lenNoise
        sigma = sigmas(j);
        
        fprintf('Dataset %s, sigma %d...\n', dataset, sigma);
        generate_sigmas(dataset, sigma);
    end
end