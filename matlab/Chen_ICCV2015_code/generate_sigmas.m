function generate_sigmas(dataset, sigma)
%dataset='bsd500'; % 'kodak', 'mcm', 'bsd500'
%sigma = 5; % 5, 15, 25

dataDir = sprintf('../../data/%s_sigma%d/', dataset, sigma);
outputDir = sprintf('../../data/chen/%s_sigma%d/', dataset, sigma);

mkdir(outputDir);

if strcmp(dataset, 'mcm')
    fileExt = '*.tif';
elseif strcmp(dataset, 'kodak')
    fileExt = '*.png'; 
elseif strcmp(dataset, 'bsd500')
    fileExt = '*.png'; 
end

files = dir(fullfile(dataDir, fileExt)); 

% sort filenames in numerical order
filenames = cell(length(files), 1); 
for i = 1 : length(files) 
    fileName = strcat(dataDir, files(i,1).name); 
    filenames{i} = fileName; 
end



for i = 1 : length(filenames)
    im_noisy = imread(filenames{i});
    estsigma =NoiseEstimation(im_noisy, 8);
    fout = fopen(sprintf('%s%s_chen.txt', outputDir, files(i,1).name), 'w');
    fprintf(fout, '%f', estsigma);
    fclose(fout);
end

end % function
