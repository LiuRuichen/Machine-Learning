data = load('dataset7.txt');
X = data(:, [1, 2]); 
y = data(:, 3);
pos = find(y == 1);
neg = find(y == 0);
scatter(X(pos,1),X(pos,2),15,'r');
hold on;
scatter(X(neg,1),X(neg,2),15,'b');
hold on;
% axis([-2 7 0 8]);
axis equal;

X_pos = X(pos,:);
X_neg = X(neg,:);

X_pos1 = X_pos;
X_neg1 = X_neg;

[X_pos, mu_pos] = fN(X_pos1);
[X_neg, mu_neg] = fN(X_neg1);
% plot(mu_pos(1),mu_pos(2),'k+','LineWidth',2,'MarkerSize',7);
% hold on;
% plot(mu_neg(1),mu_neg(2),'k+','LineWidth',2,'MarkerSize',7);



% Within-class divergence matrix£¬(2*60)*(60*2) = (2*2)
Sw = X_pos' * X_pos + X_neg' * X_neg;

% Inter-class divergence matrix£¬(2*1)*(1*2) = (2*2)
Sb = (mu_pos-mu_neg)' * (mu_pos-mu_neg);

[V, D] = eig(inv(Sw) * Sb);
SSS = inv(Sw) * Sb;
[D_sort,index] = sort(diag(D),'descend');
D_sort = D_sort(index);
V_sort = V(:,index);
V = V_sort;

omega = V(:,1);

% omega = omega/norm(omega);

k = omega(2,1)/omega(1,1);

% X1 = 10:0.01:40;
X1 = -2:0.01:1.5;
Y1 = k*X1;
plot(X1,Y1,'k');
axis equal;

projection_pos = omega' * X_pos1';
projection_neg = omega' * X_neg1';

recover_pos = omega * projection_pos;
recover_neg = omega * projection_neg;

% draw points and their projections respectively
for i = 1:size(recover_pos,2)
    scatter(recover_pos(1,i),recover_pos(2,i),15,'r');
    
    drawLine(recover_pos(:,i), X_pos1(i,:), '--k', 'LineWidth', 1);
end

for i = 1:size(recover_neg,2)
    scatter(recover_neg(1,i),recover_neg(2,i),15,'b');
    drawLine(recover_neg(:,i), X_neg1(i,:), '--k', 'LineWidth', 1);
end

