function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%% Implementation of Part 1
% add x0
% m = 5000, n = 400, K = 10, h = 25
% X = 5000 * 401 = m * (n+1)
a1 = [ones(m, 1) X];

% a2 = 5000 * 25
% Theta1 = 25 * 401
z2 = a1 * Theta1';
a2 = sigmoid(z2);

% a2 = 5000 * 26
a2 = [ones(m, 1) a2];

% hyphothesis = a3 = 5000 * 10 = m * K
% Theta2 = 10 * 26
z3 = a2 * Theta2';
a3 = sigmoid(z3);
hyphothesis = a3;

K = size(hyphothesis, 2);
new_y = zeros(m, K);

for i=1:m
    new_y(i, y(i)) = new_y(i, y(i)) + 1;
end

J = 0;

% new_y(i) = 1 * K, hyphothesis(i) = 1 * K
for i=1:m
    tmp = new_y(i, :) * log(hyphothesis(i, :))' + (1 - new_y(i, :)) * log(1 - hyphothesis(i, :))';
    J = J + tmp;
end
J = (-1 / m) * J;

% Note that you should not be regularizing the terms that correspond to the
% bias
regularization_term = (lambda / (2 * m)) * ((sum(sum(Theta1(:, 2:end) .^ 2))) + (sum(sum(Theta2(:, 2:end) .^ 2))));
J = J + regularization_term;
%% Implementation of Part2

% In our case, small delta(i) must be m * (# of units in Layer l)
% Because, we calculate all cases(# = m) together.(That is, we summarize one
% iteration's information to one row. Hence, Because there is total m iterations, We should be get m row's)

% ThetaX, DeltaX, ThetaX_grad have same demension.

% new_y = m * K, a3 = m * K
% d3 = m * K
d3 = a3 - new_y;

% Theta2(, 2:end) = K * h
% d2 = m * h
d2 = (d3 * Theta2(:, 2:end)) .* sigmoidGradient(z2);

% a1 = m * (n + 1)
% Delta1 = h * (n + 1) 
Delta1 = d2' * a1;

% Delta2 = K * (h + 1)
Delta2 = d3' * a2;

Theta1_grad = (1 / m) * Delta1;
Theta2_grad = (1 / m) * Delta2;

%% Implementation of Part3

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda / m) * Theta1(:, 2:end));
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda / m) * Theta2(:, 2:end));
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
