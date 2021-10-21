data = load('dataset9.txt');
K = 2;  %分类个数
X = data(:, 1:3); 
y = data(:, 4);

for i = 1:size(X,1)
    if y(i) == 0
        X(i,1) = X(i,1) + 4;
        X(i,2) = X(i,2) - 4;
    end
end

%% 样本可视化
for i = 1:K
    t = find(y == i-1);  %第i类的行号,column vector
    if i == 1
        scatter3(X(t,1),X(t,2),X(t,3),15,'r');
        hold on;
    else
        scatter3(X(t,1),X(t,2),X(t,3),15,'g');
        hold on;
    end
end
axis([-20 0 0 20 0 10]);
xlabel('x');
ylabel('y');
zlabel('z');

%% 计算类内散度矩阵
Mu = zeros(K,size(X,2));
m = zeros(1,K);
Sw = zeros(size(X,2),size(X,2));  % 2*2
for i = 1:K
   t = find(y == i-1);  % 第i类的行号,column vector 
   m(i) = size(t,1);  %该类中有多少样本
   X_type = X(t,:);  %X矩阵中提取出第i类的行
   [X_type_norm, Mu(i,:)] = fN(X_type);
   Sw = Sw + X_type_norm' * X_type_norm;  % 3*3的矩阵
end

%% 计算类间散度矩阵
Sb = (Mu(1,:)-Mu(2,:))' * (Mu(1,:)-Mu(2,:));  % (3*1) * (1*3) = (3*3)

%% 求解omega
[V, D] = eig(inv(Sw) * Sb);
[D_sort,index] = sort(diag(D),'descend');
D_sort = D_sort(index);
V_sort = V(:,index);
V = V_sort;
omega = V(:,1:2);

%% 在三维空间显示投影点
new_table = [];
figure;
for i = 1:K
   t = find(y == i-1);  % 第i类的行号,column vector 
   projection = omega' * X(t,:)';  % (2*3) * (3*类内样本数) = (2*类内样本数)，降维
   X_type = X(t,:);
   recover = omega * projection;  % (3*2) * (2*类内样本数) = (3*类内样本数)，复原
   new_table = [new_table;[projection' y(t,:)]];
   if i == 1
       for j = 1:size(projection,2)
           p1 = scatter(projection(1,j),projection(2,j),15,'r');
           hold on;
       end
   else
       for j = 1:size(projection,2)
           p2 = scatter(projection(1,j),projection(2,j),15,'g');
           hold on;
       end
   end  
end

%% 逻辑回归进行分类

X = new_table(:,1:2);
y = new_table(:,3);

[m, n] = size(X);

pos = find(y == 1);
neg = find(y == 0);
X_pos = X(pos,:);
X_neg = X(neg,:);

% p1 = scatter(X_pos(:,1),X_pos(:,2),'g');
% p2 = scatter(X_neg(:,1),X_neg(:,2),'r');

X = [ones(m, 1) X];

initial_theta =zeros(n + 1, 1);

option = optimset('GradObj', 'on', 'MaxIter', 400);

[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, option);

x = -15:0.01:-5;
y = (theta(1,1)+theta(2,1)*x)/(-theta(3,1));
p3 = plot(x,y,'k');

axis([-20 5 -12 6]);
legend([p1,p2,p3],{'0-class','1-class','decision boundary'});

%% 预测
A = [-10 + (0-(-10))*rand(10,1) 0 + (10-0)*rand(10,1) 0 + (10-0)*rand(10,1) ones(10,1)];
B = [-16+ (-6-(-16))*rand(10,1) 6 + (16-6)*rand(10,1) 0 + (10-0)*rand(10,1) zeros(10,1)];
data = [A; B];
% data = load('dataset9.txt');
K = 2;  %分类个数
X = data(:, 1:3); 
y = data(:, 4);

right = 0;
for i = 1:size(X,1)
    X_projection = X(i,:) * omega;
    X_projection = [1 X_projection];
    res = X_projection * theta;    
    if sigmoid(res) > 0.5
        if y(i) == 1
            right = right + 1;
        end
    elseif sigmoid(res) < 0.5
        if y(i) == 0
            right = right + 1;
        end
    end
end
fprintf('The accuracy is %g.\n',right/size(X,1));