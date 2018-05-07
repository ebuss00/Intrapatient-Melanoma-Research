function [xOverlap, yOverlap, xCentroids, yCentroids] = overlap(BW, I)
% numberOfColorChannels should be = 1 for a gray scale image, and 3 for an RGB color image.
[rows, columns, ~] = size(I);

% Label the image
labeledImage = logical(BW);

% Make the measurements
props = regionprops(labeledImage, 'Centroid', 'Orientation');
%xCentroid = props{1, 'Centroid'};
%yCentroid = props{2, 'Centroid'};

%centroids = [props.Centroid];
%xCentroids = centroids(1:2:end);
%yCentroids = centroids(2:2:end);

xCentroid = props.Centroid(1);

yCentroid = props.Centroid(2);

disp("xCentroid: " + xCentroids);
disp("yCentroid: " + yCentroids);

% Find the half way point of the image.
middlex = columns/2;
middley = rows/2;

% Translate the image
deltax = middlex - xCentroid;
deltay = middley - yCentroid;
binaryImage = imtranslate(BW,  [deltax, deltay]);

% Rotate the image
angle = -props.Orientation;
rotatedImage = imrotate(binaryImage, angle, 'crop');

topArea = sum(sum(binaryImage(1:rows/2,:)));
bottomArea = sum(sum(binaryImage(rows/2+1:end,:)));
areaDifference = abs(topArea - bottomArea);

flipped = flipud(rotatedImage);
overlapped = flipped & rotatedImage;
nonOverlapped = xor(flipped, rotatedImage);
%disp("NO: " + nonOverlapped)
sumOverlap = sum(overlapped(:));
sumNonOverlap = sum(nonOverlapped(:));
xOverlap = sumOverlap/(sumOverlap + sumNonOverlap);

%disp(xOverlap);

flipped2 = fliplr(rotatedImage);
overlapped2 = flipped2 & rotatedImage;
nonOverlapped2 = xor(flipped2, rotatedImage);
sumOverlap2 = sum(overlapped2(:));
sumNonOverlap2 = sum(nonOverlapped2(:));
yOverlap = sumOverlap2/(sumOverlap2 + sumNonOverlap2);

end
