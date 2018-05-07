function [BW4, I] = segmenter(I)
I= imgaussfilt(I,16);
gray = rgb2gray(I);
level = graythresh(gray);
BW = imbinarize (gray, level);
BW2 = imcomplement(BW);
BW3 = bwareaopen(BW2, 600);
BW4 = imfill(BW3,'holes');


I(repmat(BW,[1 1 3])) = 0;
end
