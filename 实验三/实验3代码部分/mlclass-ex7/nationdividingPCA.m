clear ; close all; clc
[X,Y,Z] = xlsread('data.xlsx');
XX = X;
X1 = X(:,2)./X(:,1);
X2 = [X1 X(:,3)];
X = X2;
scatter(X(:,1),X(:,2),15,'k');
xlabel('GDP per capita');
ylabel('Population living in slums(% of population)');
hold on;
[m,n] = size(X);

% 先降维再聚类
[U, S] = pca(X);
K = 1;  %降为2维
Z1 = projectData(X, U, K);
X_rec = recoverData(Z1, U, K);

K1 = 4;  %划分类的个数
max_iters = 1000;  %迭代个数
initial_centroids = kMeansInitCentroids(Z1, K1);
[centroids, idx] = runkMeans(Z1, initial_centroids, max_iters);

level = cell(K1,m);
ct = ones(1,K1);
for i = 1:m
    level(idx(i),ct(idx(i))) = Z(i+1,1);
    ct(idx(i)) = ct(idx(i)) + 1;
end
% % 
% 
figure;
palette = hsv(K1 + 1);
colors = palette(idx, :);

% Plot the data
scatter(X(:,1), X(:,2), 15, colors);
hold on;
scatter(X_rec(:,1), X_rec(:,2), 15, colors);
hold on;
k = U(2,1)/U(1,1);%(X_rec(2, 2)-X_rec(1, 2))/(X_rec(2, 1)-X_rec(1, 1));
b = X_rec(1, 2)-k*X_rec(1, 1);
x_plot = 0:0.001:90000;
y_plot = k*x_plot + b;
plot(x_plot,y_plot,'k');
xlabel('GDP per capita');
ylabel('Population living in slums(% of population)');
for i = 1 : K1
    fprintf('第%g类国家有：',i);
    GDP_per = 0;
    slum_por = 0;
    for j = 1:ct(i)-2
        y = level{i,j};
        fprintf('%s, ',y);
        id = find(strcmp(Z, y));
        GDP_per = GDP_per + X(id-1,1);
        slum_por = slum_por + X(id-1,2);
        if rem(j,10) == 0
            fprintf('\n');
        end
    end
    y = level{i,ct(i)-1};
    fprintf('%s, ',y);
    centroid_recovered = recoverData(centroids,U,K);
    fprintf('共计%g个, 该类国家的人均GDP为%g, 居住在贫民窟的人占总人口的比例为%g\n',...
             ct(i)-1,...
             GDP_per/(ct(i)-1),...
             slum_por/(ct(i)-1));
end
% 
ct = zeros(K1);
for i = 1:m
    ct(idx(i)) = ct(idx(i)) + 1;
end
for i = 1:K1
    fprintf('%g ',ct(i));
end
fprintf('\n');