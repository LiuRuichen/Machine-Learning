clear ; close all; clc
[X,Y,Z] = xlsread('data.xlsx');
XX = X;
X1 = X(:,2)./X(:,1);
X2 = [X1 X(:,3)];
X = X2;
scatter(X(:,1),X(:,2),15,'k');
xlabel('GDP per capita');
ylabel('Population living in slums(% of population)');
% zlabel('Slummers(% of population)');
hold on;
[m,n] = size(X);

% mu = mean(X);  %��������ƽ��ֵ���õ�ÿһ��ά�ȵ�ƽ��ֵ
% X = bsxfun(@minus, X, mu);   %��ȥ(minus)ƽ��ֵ
% 
% sigma = std(X);  %������ñ�׼��õ�ÿһ��ά�ȵı�׼��
% X = bsxfun(@rdivide, X, sigma);  %����(divide)��׼��

%% ֱ�Ӿ���
% X = X(:,1:2);
K = 4;  %������ĸ���
max_iters = 1000;  %��������
initial_centroids = kMeansInitCentroids(X, K);
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

level = cell(K,m);
ct = ones(K);
for i = 1:m
    level(idx(i),ct(idx(i))) = Z(i+1,1);
    ct(idx(i)) = ct(idx(i)) + 1;
end

figure;

plotDataPoints(X,idx,K);
xlabel('GDP per capita');
ylabel('Population living in slums(% of population)');
for i = 1 : K
    fprintf('��%g������У�',i);
    for j = 1:ct(i)-2
        y = level{i,j};
        fprintf('%s, ',y);
        if rem(j,10) == 0
            fprintf('\n');
        end
    end
    y = level{i,ct(i)-1};
    fprintf('%s, ',y);
%     fprintf('����%g��, ������ҵ��˾�GDPΪ%g, ��ס��ƶ��ߵ���ռ���˿ڵı���Ϊ%g\n',...
%             ct(i)-1, ...
%             centroids(i,2)/centroids(i,1), ...
%             centroids(i,3));
    fprintf('����%g��, ������ҵ��˾�GDPΪ%g, ��ס��ƶ��ߵ���ռ���˿ڵı���Ϊ%g\n',...
            ct(i)-1, ...
            centroids(i,1), ...
            centroids(i,2));
end
I = find(idx == i);

ct = zeros(K);
for i=1:m
    ct(idx(i)) = ct(idx(i)) + 1;
end
for i = 1:K
    fprintf('%g ',ct(i));
end
fprintf('\n');

    

