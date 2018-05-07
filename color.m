
function [features] = color(segmented)
%mole = im2double(segmented);
% Perform any segmentation operation here - set non-mole pixels to [0 0 0]

% Flatten vector for covariance computation
flat_mole = reshape(segmented,[],3);
% Throw away any 0 pixels
no_zeros = flat_mole(sum(flat_mole,2) ~= 0,:);
% Compute covariance matrix
covariance_matrix = cov(no_zeros);
% Extract non-redundant features to a vector
features = covariance_matrix(triu(ones(3,3) > 0));

end