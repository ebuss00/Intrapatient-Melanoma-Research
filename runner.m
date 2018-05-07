files = dir('**/*.JPG');
json = dir('**/*.json');
folder = files.name;
json2 = json.name;
%disp(json2);
[m,~] = size(files);
C = cell(m,18);
%matrix = zeros(m,18);


%% Batch processor

for k = 1:numel(files)
   name = files(k).name;
   disp(name);
   I = imread(name);
   features = final_draft2(I);
   %matrix = [matrix; features];
   %old = files(k).name;
   file = json(k).name;
   data = loadjson(file);
   aTable = struct2table(data); 
   string = aTable{1, 'meta'};
   bTable = struct2table(string);
   string2 = bTable{1, 'clinical'};
   diagnosis = string2.benign_malignant;
   diagnosis2 = str2double(diagnosis);
   %disp(diagnosis);
   features = num2cell(features);
   diagnosis = cellstr(diagnosis);
   C(k,1:18) = [features diagnosis];
   %matrix(k,:)= horzcat(features, diagnosis2);

end
disp(C);
cell2csv('text.csv',C,',');
   %csvwrite('new.csv',matrix);

