function TH=lpsolvefun(input)

% Note: this is to write lpsolveopt.m into a function form
% and to be used by macrosearch.m
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% solve the optimal solution 
%
% Date: 04/24/2013
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%input = [6,7,7,12,8];
%input = [7,7,6,9,11];

mu = input(1:3);
b = zeros(1,3);
b(2:3) = input(4:5);


%% using X = linsolve(A,B) to find the solution

dimX = (b(2)+2) * (b(3)+2) - 1; 
vectorB = zeros(dimX,1); % denote p(0,0),p(0,1),...,p(0,b3+1), p(1,0),...p(1,b3+1),...

matrixA = zeros(dimX);

% see Page 189 of Buzacott and Shanthikumar (1993) stochastic models of
% manufacturing systems

% 1
matrixA(1,1) = mu(1);
matrixA(1,2) = -mu(3);

% 2
for j = 1:b(3)
    matrixA(j+1,j+1) = mu(1)+mu(3);
    matrixA(j+1,b(3)+j+2) = -mu(2);
    matrixA(j+1,j+2) = -mu(3);
end

% 3
matrixA(b(3)+2,b(3)+2) = mu(1)+mu(3);
matrixA(b(3)+2,2*b(3)+3) = -mu(2);

% 4
for i = 1:b(2)
    matrixA(i*(b(3)+2)+1,i*(b(3)+2)+1) = mu(1)+mu(2);
    matrixA(i*(b(3)+2)+1,(i-1)*(b(3)+2)+1) = -mu(1);
    matrixA(i*(b(3)+2)+1,i*(b(3)+2)+2) = -mu(3);
end

% 5
for i = 1:b(2)
    for j = 1:b(3)
        matrixA(i*(b(3)+2)+1+j,i*(b(3)+2)+1+j) = mu(1)+mu(2)+mu(3);
        matrixA(i*(b(3)+2)+1+j,(i-1)*(b(3)+2)+1+j) = -mu(1);
        matrixA(i*(b(3)+2)+1+j,(i+1)*(b(3)+2)+j) = -mu(2);
        matrixA(i*(b(3)+2)+1+j,i*(b(3)+2)+2+j) = -mu(3);
    end
end

% 6
for i=1:(b(2)-1)
    matrixA((i+1)*(b(3)+2),(i+1)*(b(3)+2)) = mu(1)+mu(3);
    matrixA((i+1)*(b(3)+2),i*(b(3)+2)) = -mu(1);
    matrixA((i+1)*(b(3)+2),(i+2)*(b(3)+2)-1) = -mu(2);
end

% 7
matrixA((b(2)+1)*(b(3)+2),(b(2)+1)*(b(3)+2)) = mu(3);
matrixA((b(2)+1)*(b(3)+2),b(2)*(b(3)+2)) = -mu(1);
matrixA((b(2)+1)*(b(3)+2),(b(2)+2)*(b(3)+2)-1) = -mu(2);

% 8
matrixA((b(2)+1)*(b(3)+2)+1,(b(2)+1)*(b(3)+2)+1) = mu(2);
matrixA((b(2)+1)*(b(3)+2)+1,b(2)*(b(3)+2)+1) = -mu(1);
matrixA((b(2)+1)*(b(3)+2)+1,(b(2)+1)*(b(3)+2)+2) = -mu(3);

% 9
for j = 1:(b(3)-1)
    matrixA((b(2)+1)*(b(3)+2)+1+j,(b(2)+1)*(b(3)+2)+1+j) = mu(2)+mu(3);
    matrixA((b(2)+1)*(b(3)+2)+1+j,b(2)*(b(3)+2)+1+j) = -mu(1);
    matrixA((b(2)+1)*(b(3)+2)+1+j,(b(2)+1)*(b(3)+2)+2+j) = -mu(3);
end

% % 10 
% matrixA((b(2)+1)*(b(3)+2)+1+b(3),(b(2)+1)*(b(3)+2)+1+b(3)) = mu(2)+mu(3);
% matrixA((b(2)+1)*(b(3)+2)+1+b(3),b(2)*(b(3)+2)+1+b(3)) = -mu(1); 



%%
matrixA((b(2)+1)*(b(3)+2)+1+b(3),:) = 1;
% rank(matrixA) % should = dimX - 1; 
vectorB(dimX,1) = 1;

% disp([size(matrixA), size(vectorB)]);
% solve it!
X = linsolve(matrixA,vectorB);
    
    
% the optimal throughtput
sumProb = X(1);
for i = 1:(b(2)+1)
    sumProb = sumProb + X(i*(b(3)+2)+1);
end
    
TH = mu(3)*(1-sumProb); 
    