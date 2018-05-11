function [B] = add_similarities(A)

labels = A(:,end);
A = A(:,1:end-1);

% Normalize each feature
mus = mean(A);
stds = std(A);
A = A - mus;
A = A./stds;

features = [];
M = size(A,1);
for i = 1:M
  distances = [];
  for j = 1:M
    if (j ~= i)
      distances = [distances; norm(A(i,:)-A(j,:), 2)];
    end
  end
  features = [features; min(distances)];
end

B = [A,features,labels];


end
