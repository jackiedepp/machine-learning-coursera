function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

%num_iters = 10;

m = length(y);
J_history = zeros(num_iters, 1);
h = sigmoid(X * theta);

for iter = 1:num_iters

    theta = theta - (alpha/m) * X' * (h - y);
    h = sigmoid(X * theta);
    J_history(iter) = (1 / m) * (-y' * log(h) - (1-y)' * log(1-h));

    %[J_history(iter), grad] = costFunction(theta, X, y);
end

end
