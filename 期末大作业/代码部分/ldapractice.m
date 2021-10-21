data = load('dataset3.txt');
K = 2;  %分类个数
X = data(:, [1, 2]); 
y = data(:, 3);

%% 可视化
for i = 1:K
    t = find(y == i-1);  %第i类的行号,column vector
    if i == 1
%         plot(X(t,1),X(t,2),'ko','LineWidth',2,'MarkerSize',7);
        scatter(X(t,1),X(t,2),15,'r');
        hold on;
    else
%         plot(X(t,1),X(t,2),'k+','LineWidth',2,'MarkerSize',7)
% %         plot(X(t,1),X(t,2),'k-','LineWidth',2,'MarkerSize',7);
        scatter(X(t,1),X(t,2),15,'g');
        hold on;
    end
end
axis([0 1 0 0.8]);
axis equal;

%% 计算类内散度矩阵
Mu = zeros(K,size(X,2));
m = zeros(1,K);
Sw = zeros(size(X,2),size(X,2));  % 2*2
for i = 1:K
   t = find(y == i-1);  % 第i类的行号,column vector 
   m(i) = size(t,1);  %该类中有多少样本
   X_type = X(t,:);  %X矩阵中提取出第i类的行
   [X_type_norm, Mu(i,:)] = fN(X_type);
   Sw = Sw + X_type_norm' * X_type_norm;
end

%% 计算类间散度矩阵
Sb = (Mu(1,:)-Mu(2,:))' * (Mu(1,:)-Mu(2,:));

%% 求解omega
[V, D] = eig(inv(Sw) * Sb);
[D_sort,index] = sort(diag(D),'descend');
D_sort = D_sort(index);
V_sort = V(:,index);
V = V_sort;
omega = V(:,1);

%% 显示投影
% omega = omega/norm(omega);

k = omega(2,1)/omega(1,1);

X1 = 0:0.01:1;
Y1 = k*X1;
plot(X1,Y1,'k');

for i = 1:K
   t = find(y == i-1);  % 第i类的行号,column vector 
   projection = omega' * X(t,:)';  % (1*2) * (2*类内样本数) = (1*类内样本数)，降维
   X_type = X(t,:);
   recover = omega * projection;  % (2*1) * (1*类内样本数) = (2*类内样本数)，复原
   if i == 1
       for j = 1:size(recover,2)
           scatter(recover(1,j),recover(2,j),15,'r');
           drawLine(recover(:,j), X_type(j,:), '--k', 'LineWidth', 0.5);
       end
   elseif i == 2
       for j = 1:size(recover,2)
           scatter(recover(1,j),recover(2,j),15,'g');
           drawLine(recover(:,j), X_type(j,:), '--k', 'LineWidth', 0.5);
       end
   else 
       for j = 1:size(recover,2)
           scatter(recover(1,j),recover(2,j),15,'b');
           drawLine(recover(:,j), X_type(j,:), '--k', 'LineWidth', 0.5);
       end
   end
end