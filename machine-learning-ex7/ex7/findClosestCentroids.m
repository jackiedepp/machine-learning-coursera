function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
m = size(X, 1)
for i = 1: m
  %% suppose the first one is minimize one
  %x_i = X(i,:);
  %k_1 = centroids(1,:);
  %min = x_i - k_1;
  %min = min' * min;
  %idx(i) = 1;

  %for j = 2: K
  %  k_j = centroids(j,:);
  %  minor = x_i - k_j;
  %  minor = minor' * minor;
  %  if minor < min
  %    min = minor;
  %    idx(i) = j;
  %  end
  %end
  x_i = X(i, :);
  k_i = centroids - x_i;
  y = k_i .* k_i;
  y = sum(y, 2);
  [dump, idx(i)] = min(y);
end





% =============================================================

end
