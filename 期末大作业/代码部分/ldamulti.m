data = load('dataset2.txt');
K = 3;  %分类个数
X = data(:, [1, 2]); 
y = data(:, 3);
for i = 1:size(X,1)
    if y(i) == 0
        X(i,1) = X(i,1) + 11;
        X(i,2) = X(i,2) - 11;
    elseif y(i) == 2
        X(i,1) = X(i,1) - 7;
        X(i,2) = X(i,2) + 7;  
    end
end


%% 样本可视化
subplot(1,2,1);
for i = 1:K
    t = find(y == i-1);  %第i类的行号,column vector
    if i == 1
        scatter(X(t,1),X(t,2),15,'r');
        hold on;
    elseif i == 2
        scatter(X(t,1),X(t,2),15,'g');
        hold on;
    else
        scatter(X(t,1),X(t,2),15,'b');
        hold on;
    end
end
axis([20 80 5 50]);


%% 计算全局散度矩阵
% 规范化, X是31*2的矩阵
[X_norm, mu] = fN(X);
St = X_norm' * X_norm;  %(2*31) * (31*2) = (2*2)

%% 计算类内散度矩阵
Mu = zeros(K,size(X,2));
m = zeros(1,K);
Sw = zeros(size(X,2),size(X,2));  % 2*2
for i = 1:K
    % column vector which records the row index of ith type
    t = find(y == i-1);  
    m(i) = size(t,1);  % sample size of the ith type
    X_type = X(t,:);  % extract the samples in ith type
    [X_type_norm, Mu(i,:)] = fN(X_type);
    Sw = Sw + X_type_norm' * X_type_norm;
end

%% 计算类间散度矩阵
% Sb = St - Sw;
Sb = zeros(size(X,2),size(X,2));
for i = 1:K
    Sb = Sb + (mu - Mu(i,:))' * (mu - Mu(i,:)) * m(i);
end

%% 求解omega
[V, D] = eig(inv(Sw) * Sb);
[D_sort,index] = sort(diag(D),'descend');
D_sort = D_sort(index);
V_sort = V(:,index);
V = V_sort;
omega = V(:,1);

%% 在一维空间显示投影点
subplot(1,2,2);
for i = 1:K
    t = find(y == i-1);  %第i类的行号,column vector
    if i == 1
        scatter(X(t,1),X(t,2),15,'r');
        hold on;
    elseif i == 2
        scatter(X(t,1),X(t,2),15,'g');
        hold on;
    else
        scatter(X(t,1),X(t,2),15,'b');
        hold on;
    end
end
axis([-20 80 -30 50]);

k = omega(2,1)/omega(1,1);

X1 = -20:0.01:40;
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
