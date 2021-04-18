%-- Ucitavanje podataka--
BodyData = load('bodydata.csv');

Features = BodyData(:,1:16);
Classes = BodyData(:,17);

Features_names = {'Razmak izmedju ramena','Opseg ramena','Opseg prsa','Opseg struka','Opseg struka oko pupka','Opseg bokova','Opseg bedra','Opseg bicepsa','Opseg podlaktice','Opseg koljena ispod casice' ,'max opseg lista' ,'max opseg gleznja' ,'opseg zapesca' ,'Starost' ,'Tezina' ,'Visina' ,'spol'};


%-- PCA--
% --> I)
Features_PCA = Features(:,[1 8 16]);
[coeff, score, latent, tsquared, explained, mu] = pca(Features_PCA);

PCA1 = figure(1);

BodyData_mean = mean(Features_PCA);

endpoint1=coeff(:,1)'*15 + BodyData_mean; 
endpoint2=coeff(:,2)'*15 + BodyData_mean; 
endpoint3=coeff(:,3)'*15 + BodyData_mean; 

scatter3(Features_PCA(:,1),Features_PCA(:,2),Features_PCA(:,3),'b');
hold on;
plot3([mu(1) endpoint1(1)],[mu(2) endpoint1(2)],[mu(3) endpoint1(3)],...
    'LineWidth', 2, 'Color' , 'g');
hold on
plot3([mu(1) endpoint2(1)],[mu(2) endpoint2(2)],[mu(3) endpoint2(3)],...
    'LineWidth', 2, 'Color' , 'r');
hold on
plot3([mu(1) endpoint3(1)],[mu(2) endpoint3(2)],[mu(3) endpoint3(3)],...
    'LineWidth', 2, 'Color',  'm');
legend('Znacajke','Razmak izmedu ramena','Opseg biecpsa','Visina');
title('Stvarni podaci i smjerovi projekcije');
saveas(PCA1,sprintf('PCA 1.jpg'));

% --> II)
PCA2 = figure(2);

biplot(coeff,'Scores',score,'Varlabels',Features_names([1 8 16]));
title('Projicirani podaci i smjerovi projekcije');
saveas(PCA2,sprintf('PCA 2.jpg'));

% --> III)
PCA3 = figure(3);
pareto(explained, Features_names([1 8 16]));
saveas(PCA3,sprintf('PCA 3.jpg'));



%-- Analiza komponenti
Features_AK=Features(:,1:13);
[coeff, score, latent, tsquared, explained, mu]=pca(Features_AK);

AK1 = figure(4);
pareto(explained, Features_names(1:13));
saveas(AK1,sprintf('AK 1.jpg'));



%-- Opseg blokova i težina (k-means, Gaussova mjesavina)--
Features_KMI = Features(:,[6 15]);
eva = evalclusters(Features_KMI,'KMEANS','silhouette','KList',1:8);
K = eva.OptimalK; % Optimal k = 2

idxKM = kmeans(Features_KMI,K,'Distance','cityblock','Display','iter');
KM1 = figure(5);
[slih,h] = silhouette(Features_KMI,idxKM,'cityblock');
h = gca;
h.Children.EdgeColor=[.8 .8 1];
xlabel 'Silhouette';
ylabel 'Klaster';
saveas(KM1,sprintf('KM 1-1.jpg'));


KM2 = figure(6);
gscatter(Features_KMI(:, 1), Features_KMI(:, 2), kmeans(Features_KMI, K));
title('K-means');
xlabel('Opseg bokova');
ylabel('Težina');
legend('klaster 1','klaster 2');
saveas(KM2,sprintf('KM 1-2.jpg'));


GM = figure(7); 
idxGM = fitgmdist(Features_KMI,K);
gscatter(Features_KMI(:, 1), Features_KMI(:, 2), cluster(idxGM, Features_KMI));
title('Gausova mješavina');
xlabel('Opseg bokova');
ylabel('Težina');
legend('klaster 1','klaster 2');
saveas(GM,sprintf('GM 1-1.jpg'));

% 
% %-- Opseg prsa, opeseg struka i težina (k-means, Gaussova mjesavina)--
% Features_KMI = Features(:,[3 4 15]);
% eva = evalclusters(Features_KMI,'KMEANS','silhouette','KList',1:8);
% K = eva.OptimalK; % Optimal k = 2
% 
% idxKM = kmeans(Features_KMI,K,'Distance','cityblock','Display','iter');
% KM1 = figure(8);
% [slih,h] = silhouette(Features_KMI,idxKM,'cityblock');
% h = gca;
% h.Children.EdgeColor=[.8 .8 1];
% xlabel 'Silhouette';
% ylabel 'Klaster';
% saveas(KM1,sprintf('KM 2-1.jpg'));
% 
% KM2 = figure(9);
% gscatter(Features_KMI(:, 1), Features_KMI(:, 2), kmeans(Features_KMI, K));
% title('K-means');
% legend('klaster 1','klaster 2');
% saveas(KM2,sprintf('KM 2-2.jpg'));
% 
% 
% GM = figure(10); 
% idxGM = fitgmdist(Features_KMI,K);
% gscatter(Features_KMI(:, 1), Features_KMI(:, 2), cluster(idxGM, Features_KMI));
% title('Gausova mješavina');
% legend('klaster 1','klaster 2');
% saveas(GM,sprintf('GM 2-1.jpg'));



