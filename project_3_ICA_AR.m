clc;clear; close all;
path=strcat(pwd,'\AR_database_cropped\test2\');

ImageList=dir(fullfile(path,'*.bmp'));
N=size(ImageList,1)-10; %Leaving 10 for test data.
r = 0;
c = 0;
l = 0;
for i=1:2600
    fname = ImageList(i).name;
    fname = strcat(path,fname);
    Img = imread(fname);
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
for i=1:4
    CatData = Data(CatVec(:)==i,:);
    CatData = prewhiten(CatData');
    mdl = rica(CatData,6); %choosing 6 learned vectors.
    trans = transform(mdl,CatData);
   
    figure;
    for m = 1:6
    trans(:,m) = trans(:,m)/norm(trans(:,m))*norm(CatData(:,m));
    end
    R = rand(1, 6);
    fvc = R * trans';
        fvc = reshape(fvc,[r c]);
        imshow(mat2gray(fvc));

    title(['Category-' num2str(i) ': projection of randomly generated linear combination of ICA subspace']);
    ProjData = CatData'*trans;
    u = mean(ProjData, 1);
    sd = std(ProjData);
    figure;    
    for m=1:6
        samplePoint = normrnd(u, sd); %holds random point in normal distribution
        samplePoint = samplePoint * trans'; % projecting to original space
        samplePoint = reshape(samplePoint,[r c]);
        subplot(2,3,m);
        imshow(mat2gray(samplePoint));
        hold on;
    end
    title(['Category-' num2str(i) ': eigen faces from normal distribution from mean and variance']);
end
