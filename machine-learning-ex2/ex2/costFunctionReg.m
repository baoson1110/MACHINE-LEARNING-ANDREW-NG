function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
%grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

%J_normal = 0;
%J_regularization = 0;

%for i=1:m
%   J_normal = J_normal + 1/m*(-y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*(log(1-sigmoid(X(i,:)*theta))));
%end

%for j = 2:(size(X)(2))
%  J_regularization = J_regularization + lambda/(2*m)*theta(j)^2;
%end

%J = J_normal + J_regularization;

%reg_scalar = ones(length(theta),1);
%reg_scalar(1) = 0;

%TransMatrix = 1/m*X'*(sigmoid(X*theta)-y) + lambda/m*(reg_scalar.*theta);
%for i=1:size(theta)
%  grad(i) = TransMatrix(i);

h = sigmoid(X*theta);

J = 1/m*(-y'*log(h)-(1-y)'*log(1-h)) + lambda/(2*m)*sum(theta(2:size(theta)).^2);



grad = 1/m*X'*(h-y) + (lambda/m)*[0; theta(2:size(theta))];

 

% =============================================================

end
