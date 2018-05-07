function [fragIndex] = frag(BW)
stats = regionprops(BW, 'Perimeter', 'Area');
perim = [stats(:).Perimeter] ;
area = [stats(:).Area] ;
fragIndex = (4.*pi.*area)/(perim.*perim);

end