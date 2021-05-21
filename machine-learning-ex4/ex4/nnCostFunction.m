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


% add bias units to matrix X
X = [ones(m,1), X]';

% calculate the activation units of the layer 2 in the neural network (nn)

L2 = sigmoid(Theta1*X);

% add the bias unit to the Layer 2

L2 = [ones(1,m); L2];

% calculate the activation units of layer 3 (output units) in the neural network

L3 = sigmoid(Theta2*L2);

% create a output matrix corresponding to a m dimensional y vector
 
y_output = zeros(num_labels,m);

for i=1:m,
	y_output(y(i),i)=1;
end;

% cost function for neural network

J = 1/m*sum((-y_output.*log(L3)-(1-y_output).*log(1-L3))(:)) + lambda/(2*m)* (sum((Theta1(:,2:end).^2)(:))+sum((Theta2(:,2:end).^2)(:)));



% calculate delta3:

delta3 = L3 - y_output;

% calculate delta2:

z2 = Theta1*X;
delta2 = (Theta2)'*delta3.*L2.*(1-L2);

% calculate DELTA2

DELTA2 = delta3*(L2)';
DELTA1 = delta2(2:end,:)*X';

reg_Theta1 = Theta1;
reg_Theta2 = Theta2;

reg_Theta1(:,1) = 0;
reg_Theta2(:, 1) = 0;

%Theta1_grad = 1/m*DELTA1;
%Theta2_grad = 1/m*DELTA2;

Theta1_grad = 1/m*DELTA1 + lambda/m*reg_Theta1;
Theta2_grad = 1/m*DELTA2 + lambda/m*reg_Theta2;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
