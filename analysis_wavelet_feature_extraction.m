list_file = dir(fullfile('./', '*.mat'));

max_point = 593000;
min_point = 10001;
threshold_vals = 3:7;
lowest_frequency = 0.5;
nyq = 500.;

for alpha = 1:length(list_file) 
    file_name = list_file(alpha).name;
    patient_name = file_name(1:end-4);
    names{alpha} = patient_name; %#ok<SAGROW>
    disp(patient_name)

    load(strcat(patient_name, '.mat'))
    
    [chan, len] = size(data);
    for j = 1:chan  % high pass, f > 0.5 Hz
        [b, a] = butter(2, lowest_frequency/nyq, 'high');  
        data(j, :) = filtfilt(b, a, data(j, :)); %#ok<SAGROW>
        freq = [49 51];
        for i = 0:8  % notch
           [b, a] = butter(2, (freq + i * 50)/nyq, 'stop');  % notch filter 
           data(j, :) = filtfilt(b, a, data(j, :));
        end
    end
    
    %%%% here we get rid of the first 10 seconds
    data = data(:, min_point:max_point);
    len = length(data(1,:));
    %%%% four moments of the distribution
    moments = zeros(chan, 4); 
    moments(:, 1) = mean(data, 2);
    moments(:, 2) = std(data, 0, 2);
    moments(:, 3) = skewness(data, 1, 2);
    moments(:, 4) = kurtosis(data, 1, 2);
    
    %%%% fft features
    freq_list = [1 4 8 13 30 70 90 140 190 240 290 340 390 440 490];
    fft_feat = zeros(chan, length(freq_list)-1);
    fft_transf = fft(data, len, 2);
    fft_freqs = 1000. * (0:len/2)/len;  % 1000. sampling frequency
    for i = 1:length(freq_list)-1
       idx = (fft_freqs > freq_list(i)) & (fft_freqs < freq_list(i+1));
       fft_feat(:, i) = sum(fft_transf(:, idx).^2, 2);
    end
    %%%% relative energy
    relative_energy = abs(fft_feat) ./ sum(abs(fft_feat), 2);
    
    %%%% DWT entropy, CWT features, Renyi entropy
    dwtname = 'db2';
    cwtname = 'amor';
    freqs_cwt = 99;
    n_scales = wmaxlev(length(data(1,:)), dwtname);
    wave_bands = zeros(chan, 17);  % related to the scales
    max_cwt = zeros(chan, freqs_cwt);   % percentile, max, norm
    norm_cwt = zeros(chan, freqs_cwt);
    perc_cwt = zeros(chan, freqs_cwt);
    
    for i = 1:chan
        [c, l] = wavedec(data(i, :), n_scales, dwtname);
        dwt_coefs = detcoef(c, l, 1:17);
        for k = 1:length(dwt_coefs)
            wave_bands(i, k) = sum(dwt_coefs{1, k}.^2);
        end
        cwt_coefs = cwt(data(i, :), cwtname, 1000.);
        cwt_coefs = abs(cwt_coefs(1:freqs_cwt, :));
        max_cwt(i, :) = max(cwt_coefs, [], 2);
        perc_cwt(i, :) = prctile(cwt_coefs, 80, 2);
        for s = 1:freqs_cwt
            norm_cwt(i, s) = norm(cwt_coefs(s, :));
        end
    end
    relative_wave_bands = wave_bands ./ sum(wave_bands, 2);
 
    shannon_entropy = -sum(relative_wave_bands.*log(relative_wave_bands), 2);
    renyi_entropy = -log(sum(relative_wave_bands.^2, 2));
    %max val for each column, across channels
    max_val_max = max(max_cwt);  
    max_val_perc = max(perc_cwt);
    max_val_norm = max(norm_cwt);
    
    %%%% relative max, perc, norm values for each channel
    max_cwt = max_cwt ./ max_val_max;
    perc_cwt = perc_cwt ./ max_val_perc;
    norm_cwt = norm_cwt ./ max_val_norm;
    
    %%%% threshold
    
    total_energy = sum(abs(fft_feat), 2);
    idx = find(total_energy == min(total_energy));
    threshold = zeros(chan, (length(freq_list)-1)*length(threshold_vals));
    tmp = zeros(chan, len);
    count = 1;
    for f = 1:length(freq_list)-1
        % bandpass
        [b, a] = butter(2, [freq_list(f) freq_list(f+1)]/nyq, 'bandpass');
        for c = 1:chan
            tmp(c, :) = filtfilt(b, a, data(c, :));
        end
        std_low_e = std(tmp(idx, :));
        % how much time the signal stays over the threshold
        % we use the absolute value
        for t = threshold_vals
            threshold(:, count) = sum(abs(tmp) > std_low_e * t, 2)/1000.;            
            count = count + 1;
        end
    end
    
    % create the matrix containing all the features
    % moments, relative_energy, relative_wave_bands, shannon_entropy
    % renyi_entropy, max_cwt, perc_cwt, norm_cwt, threshold
    matrix_pat = cat(2, moments, relative_energy, relative_wave_bands, shannon_entropy, renyi_entropy, max_cwt, perc_cwt, norm_cwt, threshold);
    s = struct('labels', index, 'y', y, 'feat', matrix_pat);
    save(strcat((patient_name), 'Features.mat'), 's');
    
    clear data;
    clear moments;
    clear fft_transf;
    clear fft_feat;
    clear relative_energy;
    clear cwt_coefs;
    clear dwt_coefs;
    clear tmp;
    clear threshold;
    
end

