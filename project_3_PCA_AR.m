clc; clear; close all;
path=strcat(pwd,'\AR_database_cropped\test2\');

ImageList=dir(fullfile(path,'*.bmp'));
N=size(ImageList,1); %Leaving 10 for test data.
r = 0;
c = 0;
l = 0;
for i=1:260
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

%TODO: complete the following TODO ;) 
%% Applying for total_classes categories.
total_classes = 2;
total_dists = 2;
Ogmm= {};
Pgmm = {};
TestData = {};

for i=1:total_classes
    CatData = Data(CatVec(:)==i,:);
    TestData{i} = CatData(1, :);
    OverallMean = mean(CatData,1);
    for j=1:size(CatData,1)
        meanNormCatData(j,:) = CatData(j,:)- OverallMean;
    end
    
    %Qestion-1: 
    %part A: Estimating the distributions in the original space.
    [a, Ogmm{i}, b] = mixGaussEm(CatData', total_dists);
    
    %Part B: Estimating the distributions in the PCA space
    [cSigma,PCAspace, EValues] = PCACal(meanNormCatData);
    ProjData = meanNormCatData*PCAspace;
    [a, Pgmm{i}, b] = mixGaussEm(ProjData', total_dists); 
    figure;
    %Qestion-2: 
    for dist_index = 1:total_dists   
    %Derive the samples from all the distributions.
        %Generate samples in the original space
        OsamplePoint = mvnrnd(Ogmm{i}.mu(:,dist_index), Ogmm{i}.Sigma(:, :, dist_index)); 
        OsamplePoint = reshape(OsamplePoint,[r c]);
        subplot(total_classes,total_dists*2,i+dist_index -1);
        imshow(mat2gray(OsamplePoint));
        %Generate samples in the PCA space
        PsamplePoint = mvnrnd(Pgmm{i}.mu(:,dist_index), Pgmm{i}.Sigma(:, :, dist_index));
        reprojPoint = PsamplePoint * PCAspace' + OverallMean;
        reprojPoint = reshape(reprojPoint,[r c]);
        subplot(total_classes,total_dists*2,i+dist_index);
        imshow(mat2gray(reprojPoint));
    end
end  

distance = [];
hits = 0;
miss = 0;

%% Qestion-1 part-C: Classification using Mahalanobis distance:
    % Calculate the distance from test sample to the all the distributions
    % in all the classes and catagorize the class based on the smallest distance.
for test_index=1:total_classes
    fprintf("Testitn with %d\n", test_index);
    Y = TestData{test_index}; 
    closest_class = inf;

    for cls_index=1:total_classes
        min_distance = inf;
        for dist_index=1:2
            var = Ogmm{cls_index}.Sigma(:, :, dist_index);
            mu = Ogmm{cls_index}.mu(:,dist_index);
            distance(dist_index) = (Y-mu')*inv(var)*(Y-mu')';
        end
        temp_min = min(distance);
        if(min_distance < temp_min)
            fprintf('Found min: changing from %d to %d\n', closest_class, cls_index );
            min_distance = temp_min;
            closest_class = cls_index;
        end
    end
    if (closest_class == test_index)
        hits = hits + 1;
    else
        miss = miss + 1;
    end
end
fprintf('Classification accuracy = %f \n', (hits/(hits+miss))*100);