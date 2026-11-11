function results = evaluate_matlab(sr_dir, hr_dir, scale, count, output_csv)
% Reproduce the historical R-channel PSNR/SSIM/FSIM evaluation convention.
% The reference SSIM/FSIM functions retain their own research-use licenses.
if nargin < 5
    error('Specify SR folder, HR folder, scale, image count and output CSV.');
end
assert(ismember(scale, [2 4 8]) && count > 0 && count == floor(count));
assert(~isfile(output_csv), 'Output CSV already exists. Choose another path.');
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'third_party', 'matlab_metrics'));
PSNR = zeros(count, 1); SSIM = zeros(count, 1); FSIM = zeros(count, 1);
for i = 1:count
    sr = double(imread(fullfile(sr_dir, sprintf('%d.png', i))));
    hr = double(imread(fullfile(hr_dir, sprintf('%d.png', i))));
    assert(isequal(size(sr), size(hr)), 'SR/HR shape mismatch.');
    sr = sr(:, :, 1); hr = hr(:, :, 1);
    h = floor(size(sr, 1) / scale) * scale;
    w = floor(size(sr, 2) / scale) * scale;
    sr = sr(1:h, 1:w); hr = hr(1:h, 1:w);
    sr = sr(scale+1:end-scale, scale+1:end-scale);
    hr = hr(scale+1:end-scale, scale+1:end-scale);
    FSIM(i) = FeatureSIM(sr, hr);
    % The original compute_psnr(..., 4) shaves four additional pixels.
    a = sr(5:end-4, 5:end-4); b = hr(5:end-4, 5:end-4);
    error_pixels = a(:) - b(:);
    PSNR(i) = 20 * log10(255 / sqrt(mean(error_pixels.^2)));
    SSIM(i) = ssim_index(a, b);
end
results = table((1:count)', PSNR, SSIM, FSIM, ...
    'VariableNames', {'Image', 'PSNR', 'SSIM', 'FSIM'});
writetable(results, output_csv);
fprintf('N=%d, scale=%d, R channel: PSNR %.8f, SSIM %.8f, FSIM %.8f\n', ...
    count, scale, mean(PSNR), mean(SSIM), mean(FSIM));
end
