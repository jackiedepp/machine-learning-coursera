function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

% C, sigma options
c_opt = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
s_opt = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
c_min = 0;
s_min = 0;

error_min = 1.0;
m = size(c_opt);
n = size(s_opt);
for i = 1:m
  for j = 1:n
    % get model
    model = svmTrain(X, y, c_opt(i), @(x1,x2) gaussianKernel(x1,x2,s_opt(j)));

    % predict error
    pred = svmPredict(model, Xval);
    error = mean(double(pred ~= yval));

    % get the min error
    if error < error_min
      error_min = error;
      c_min = i;
      s_min = j;
    end
  end
end

C = c_opt(c_min);
sigma = s_opt(s_min);



% =========================================================================

end
