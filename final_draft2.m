function [featureVect] = final_draft2(I)
%% Read image

%% Pre-processing & filtering
[BW, segmented] = segmenter (I);
imshow(BW);

%% Asymmetry
fragIndex = frag(BW);
disp(fragIndex)
%[xOverlap, yOverlap, xCentroids, yCentroids] = overlap(BW, segmented);
props = regionprops(BW, 'Centroid', 'Orientation');
%xCentroid = props{1, 'Centroid'};
%yCentroid = props{2, 'Centroid'};

centroids = [props.Centroid];
xCentroids = centroids(1:2:end);
yCentroids = centroids(2:2:end);

disp(xCentroids);
disp(yCentroids);
%disp("Percent x overlap: " + xOverlap);
%disp("Percent y overlap: " + yOverlap);

%% Shape
stats2 = regionprops(BW,'MajorAxisLength', 'MinorAxisLength', 'Eccentricity');
major = stats2.MajorAxisLength;
minor = stats2.MinorAxisLength;
eccen = stats2.Eccentricity;

   
%% Color variegation
gray = rgb2gray(I);

imshow(gray);
glcms = graycomatrix(gray);
stats3 = graycoprops(glcms,{'Contrast','Correlation', 'Energy', 'Homogeneity'});
contrast = stats3.Contrast;
correlation = stats3.Correlation;
energy = stats3.Energy;
homogeneity = stats3.Homogeneity;


%% Color
mole = im2double(segmented);

flat_mole = reshape(mole,[],3);
% Throw away any 0 pixels
no_zeros = flat_mole(sum(flat_mole,2) ~= 0,:);
% Compute covariance matrix
covariance_matrix = cov(no_zeros);
% Extract non-redundant features to a vector
features = covariance_matrix(triu(ones(3,3) > 0));
features =  features.';
A = features(1);
Z = features (2);
C = features (3);
D = features (4);
E = features (5);
F = features (6);

R = (segmented(:,:,1));
G = (segmented(:,:,2));
B = (segmented(:,:,3));

R2 = (R(R~=0));
G2 = mean(G(G~=0));
B2 = mean(B(B~=0));

meanR = mean(R2);
meanG = mean(G2);
meanB = mean(B2);

%% Classifier

%% Feature vector!
%doesn't contain similarity metric (norm of difference) yet-- waiting on
%intrapatient images

featureVect = horzcat(fragIndex, major, minor, eccen, contrast, correlation, energy, homogeneity, A, Z, C, D, E, F, meanR, meanG, meanB);
%disp(featureVect);
%isscalar(featureVect);

%struct2csv(allresults, training.txt);
end
