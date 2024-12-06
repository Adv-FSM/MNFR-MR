clc;
clear all;

% Load your dataset path
mat = load('...\dataset\HDSS\COIL20'); 
d = mat.('X');
f = mat.('Y');

K = 30; % Number of features to be selected
chi = 0; % Noise censoring threshold, chi = -3, -2, -1, 0

nd = size(d,2);
nc = size(d,1);

for i = 1:nd
   I(i) = mutualinfo(d(:,i),f); % mutual information
end
mean_1 = mean(d,1);
std_1 = std(d,1);
min_1 = min(d, [], 1);

for i = 1:nd
 sigma_1_2(i) = std_1(i)^2 - mean_1(i)^2; % ^{n}sigma_j, if max{sigma_1_2(i)} >= 0
 % sigma_1_2(i) = std_1(i)^2; % ^{n}sigma_j, if max{sigma_1_2(i)} < 0

 if sigma_1_2(i) < 1/[20*pi]
     I_nf(i) = I(i);
 elseif sigma_1_2(i) == 1/[20*pi]
     I_nf(i) = I(i);
 else
     tH(i) = 1+log10(0.1+mean_1(i)+abs(min_1(i)-chi*sqrt(sigma_1_2(i))));
     I_nf(i) = [tH(i)/(0.5*log10(2*pi*(sigma_1_2(i)))+0.5+tH(i))]*I(i);
 end
end

[tmp, idxs] = sort(-I_nf);
fea(1) = idxs(1);
idxleft = idxs(2:nd);

for k = 2:K
    nd_left = length(idxleft);
    curlastfea = length(fea);
    for i = 1:nd_left
        II(i) = mutualinfo(d(:,idxleft(i)), f);
        mi_array(idxleft(i),curlastfea) = getmultimi(d(:,fea(curlastfea)), d(:,idxleft(i)));
        RY(i) = mean(mi_array(idxleft(i), :));
        mean_2(i) = mean(d(:,idxleft(i)));
        std_2(i) = std(d(:,idxleft(i)));
        min_2(i) = min(d(:,idxleft(i)));
        sigma_2_2(i) = std_2(i)^2 - mean_2(i)^2; % ^{n}sigma_j, if max{sigma_2_2(i)} >= 0
        % sigma_2_2(i)= std_2(i)^2; % ^{n}sigma_j, if max{sigma_2_2(i)} < 0
        if sigma_2_2(i) < 1/[20*pi]
          MNFR_MR(i) = II(i)-RY(i);
        elseif sigma_2_2(i) == 1/[20*pi]
          MNFR_MR(i) = II(i)-RY(i);
        else
          tH_2(i) = 1+log10(0.1+mean_2(i)+abs(min_2(i)-chi*sqrt(sigma_2_2(i))));
          MNFR_MR(i) = [tH_2(i)/(0.5*log10(2*pi*(sigma_2_2(i)))+0.5+tH_2(i))]*II(i)-RY(i);
        end
    end
    [tmp, fea(k)] = max(MNFR_MR(1:nd_left));
    tmpidx = fea(k); 
    fea(k) = idxleft(tmpidx); 
    idxleft(tmpidx) = [];
end

fea




