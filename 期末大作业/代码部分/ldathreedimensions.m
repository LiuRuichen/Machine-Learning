data = load('dataset9.txt');
K = 2;  %�������
X = data(:, 1:3); 
y = data(:, 4);

for i = 1:size(X,1)
    if y(i) == 0
        X(i,1) = X(i,1) + 4;
        X(i,2) = X(i,2) - 4;
    end
end

%% �������ӻ�
for i = 1:K
    t = find(y == i-1);  %��i����к�,column vector
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

%% ��������ɢ�Ⱦ���
Mu = zeros(K,size(X,2));
m = zeros(1,K);
Sw = zeros(size(X,2),size(X,2));  % 2*2
for i = 1:K
   t = find(y == i-1);  % ��i����к�,column vector 
   m(i) = size(t,1);  %�������ж�������
   X_type = X(t,:);  %X��������ȡ����i�����
   [X_type_norm, Mu(i,:)] = fN(X_type);
   Sw = Sw + X_type_norm' * X_type_norm;  % 3*3�ľ���
end

%% �������ɢ�Ⱦ���
Sb = (Mu(1,:)-Mu(2,:))' * (Mu(1,:)-Mu(2,:));  % (3*1) * (1*3) = (3*3)

%% ���omega
[V, D] = eig(inv(Sw) * Sb);
[D_sort,index] = sort(diag(D),'descend');
D_sort = D_sort(index);
V_sort = V(:,index);
V = V_sort;
omega = V(:,1:2);

%% ����ά�ռ���ʾͶӰ��
new_table = [];
figure;
for i = 1:K
   t = find(y == i-1);  % ��i����к�,column vector 
   projection = omega' * X(t,:)';  % (2*3) * (3*����������) = (2*����������)����ά
   X_type = X(t,:);
   recover = omega * projection;  % (3*2) * (2*����������) = (3*����������)����ԭ
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

%% �߼��ع���з���

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

%% Ԥ��
A = [-10 + (0-(-10))*rand(10,1) 0 + (10-0)*rand(10,1) 0 + (10-0)*rand(10,1) ones(10,1)];
B = [-16+ (-6-(-16))*rand(10,1) 6 + (16-6)*rand(10,1) 0 + (10-0)*rand(10,1) zeros(10,1)];
data = [A; B];
% data = load('dataset9.txt');
K = 2;  %�������
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