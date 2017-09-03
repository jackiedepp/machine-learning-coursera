function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);


x1 = [X(:,1)];
x2 = [X(:,2)];

ori_theta = theta;
a = [];
b = [];


for iter = 1:num_iters
%for iter = 1:3
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    A = X * theta - y;
    %B = A' * X;
    B = X' * A;

    %theta(1) = theta(1) - (alpha/m) * sum(x1.*A)
    %theta(2) = theta(2) - (alpha/m) * sum(x2.*A)

    %ori_theta = ori_theta - (alpha/m) * B;
    %theta = ori_theta;
    theta = theta - (alpha/m) * B;



    %s1 = 0;
    %s2 = 0;
    %for i = 1:m
    %   s1 = s1 + (theta(1)*x1(i) + theta(2)*x2(i) - y(i))*x1(i);
    %   s2 = s2 + (theta(1)*x1(i) + theta(2)*x2(i) - y(i))*x2(i);
    %end

    %theta(1) = theta(1) - (alpha/m) * s1;
    %theta(2) = theta(2) - (alpha/m) * s2;


    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

end

iter = 1:1:num_iters;
plot(iter, J_history(iter))
end
