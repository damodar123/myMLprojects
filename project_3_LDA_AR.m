clc;clear; close all;
path=strcat(pwd,'\AR_database_cropped\test2\');

ImageList=dir(fullfile(path,'*.bmp'));
N=size(ImageList,1);
r = 0;
c = 0;
l = 0;
for i=1:260
    fname = ImageList(i).name;
    fname = strcat(path,fname);
    Img = imread(fname);
    Img = imresize3(Img,[32 32 3]);
    [r,c,l] = size(Img);
    Data(i,:) =reshape(im2double(rgb2gray(Img)), [1,r*c]);
    catname = regexp(fname,'(\d{2}).bmp','match');
    catname = catname{1};
    if(str2num(catname(1:2))>13)
        CatVec(i) = str2num(catname(1:2)) - 13 ;
    else
        CatVec(i) = str2num(catname(1:2));
    end
end

CatDataA = Data(CatVec(:)==1,:);
CatDataB = Data(CatVec(:)==2,:);
%Mean of each class
muA(1,:)=mean(CatDataA) ;
muA(2,:)=mean(CatDataB) ;
% Center the data (data-mean)
dat1=CatDataA-muA(1,:);
dat2=CatDataB-muA(2,:);
% Calculate the within class variance (SW)
covar1=dat1'*dat1 ;
covar2=dat2'*dat2 ;
sw=covar1+covar2 ;
invsw=inv(sw) ;
% calculate the across class scatter matrix (SB)
mu = mean(muA);
SB = zeros(r*c, r*c);
for cls_indx = 1:2
    mean_aligned_matrix =  muA(cls_indx,:) - mu;
    SB = SB + mean_aligned_matrix * mean_aligned_matrix';
end

datv=invsw*SB;
datSym = datv+datv.';
[V, L] = eig(datSym);
[eigValues, ind] = sort(diag(L),'descend');
eig_vecs = V(:,ind);
figure;  
%Selecting a randomly generated linear combination of eigen vector
% and reprojecting into the original space
for k=1:6
    R = rand(1, size(eig_vecs, 2));
    fvc = R * eig_vecs' + mu;
    fvc=reshape(fvc,[r c]);
    subplot(2,3,k);
    imshow(mat2gray(fvc));
    hold on;
end
%we need a minimum of class-1 eigen vectors
title('projection of first six eigen vectors into original space');


%part 2: calculating mean and covariance of data in pca subspace
ProjDataClassA = CatDataA*eig_vecs;
u = mean(ProjDataClassA, 1);
sd = std(ProjDataClassA, 1);
figure;

for m=1:6
    samplePoint = normrnd(u, sd); %holds random point in normal distribution
    samplePoint = samplePoint * eig_vecs'; % projecting to original space
    samplePoint1 = reshape(samplePoint,[r c]);
    subplot(2,3,m);
    imshow(mat2gray(samplePoint1));
    hold on;
end
title(['category:',num2str(1),'projection from normal distribution of LDA']);

ProjDataClassB = CatDataB*eig_vecs;
u = mean(ProjDataClassB, 1);
sd = std(ProjDataClassB, 1);
figure;

for m=1:6
    samplePoint = normrnd(u, sd); %holds random point in normal distribution
    samplePoint = samplePoint * eig_vecs'; % projecting to original space
    samplePoint1 = reshape(samplePoint,[r c]);
    subplot(2,3,m);
    imshow(mat2gray(samplePoint1));
    hold on;
end
title(['category:',num2str(2),'projection from normal distribution of LDA']);
