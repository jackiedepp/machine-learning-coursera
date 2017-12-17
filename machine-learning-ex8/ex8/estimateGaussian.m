function [mu sigma2 p] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X),
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
%

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

mu = (1/m) * sum(X, 1); % row vector

x_norm  = X - mu;
x_norm2 = x_norm .^ 2;
sigma2 = (1/m) * sum(x_norm2, 1); % row vector

%x = - 0.5 * (x_norm .^ 2) / sigma2;
%p = (2 * pi) ^ (-0.5) * (sigma2 .^ (-0.5)) * exp(sum(x, 2));

% compute guassian not multivariateGaussian by lizhi
p = ones(m, 1);
for i = 1:m
  for j = 1:n
    p_j = (2*pi*sigma2(j))^(-0.5) * exp(-0.5* (x_norm(i,j))^2 / sigma2(j) );
    p(i) = p(i) * p_j;
  end
end

mu = mu';
sigma2 = sigma2';

% =============================================================


end
