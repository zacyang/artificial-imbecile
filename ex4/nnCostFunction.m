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
% for 3 layers first
% input -> hidden
% caculate h
% 1.add ones in this first column
addedLeadingOnes=  [ones(size(X, 1), 1) X]
input2hidden = sigmoid(addedLeadingOnes * Theta1');
% add ones again...
outputOfLayer2=  [ones(size(input2hidden, 1), 1) input2hidden]
h=  sigmoid(outputOfLayer2 * Theta2');

% tmp = -y .* log(h)- (1 - y) .* log(1- h);
yVec = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);

cost = -yVec .* log(h) - (1 - yVec) .* log(1 - h);


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

theta1ExcludingBias = Theta1(:, 2:end);
theta2ExcludingBias = Theta2(:, 2:end);

reg1 = sum(sum(theta1ExcludingBias .^ 2));
reg2 = sum(sum(theta2ExcludingBias .^ 2));

J = (1 / m) * sum(sum(cost)) + (lambda / (2 * m)) * (reg1 + reg2);
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));
for t = 1:m;
% setp 1, comute the activations
	h1t = h(t, :)';
	a1t = addedLeadingOnes(t,:)';
	a2t = outputOfLayer2(t, :)';
	yVect = yVec(t, :)';

	d3t = h1t - yVect;
	z2t = [1; Theta1 * a1t];
    d2t = Theta2' * d3t .* sigmoidGradient(z2t);


    delta1 = delta1 + d2t(2:end) * a1t';
       % op1 is 4x5, op2 is 4x4
    delta2 = delta2 + d3t * a2t';

end;

Theta1_grad = (1 / m) * delta1;
Theta2_grad = (1 / m) * delta2;
% -------------------------------------------------------------

% regulartion

theta1ExcludingBias = Theta1(:, 2:end);
theta2ExcludingBias = Theta2(:, 2:end);

Theta1ZeroedBias = [ zeros(size(Theta1, 1), 1) theta1ExcludingBias ];
Theta2ZeroedBias = [ zeros(size(Theta2, 1), 1) theta2ExcludingBias ];

Theta1_grad = Theta1_grad + (lambda/m)* Theta1ZeroedBias;

Theta2_grad = Theta2_grad + (lambda/m)* Theta2ZeroedBias;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
