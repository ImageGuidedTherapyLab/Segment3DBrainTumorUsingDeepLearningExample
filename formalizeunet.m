% created from https://www.mathworks.com/help/images/segment-3d-brain-tumor-using-deep-learning.html
% >> ver -support
% -----------------------------------------------------------------------------------------------------
% MATLAB Version: 9.8.0.1417392 (R2020a) Update 4
% MATLAB License Number: 68666
% Operating System: Linux 4.4.0-127-generic #153-Ubuntu SMP Sat May 19 10:58:46 UTC 2018 x86_64
% Java Version: Java 1.8.0_202-b08 with Oracle Corporation Java HotSpot(TM) 64-Bit Server VM mixed mode
% -----------------------------------------------------------------------------------------------------
% MATLAB                                                Version 9.8         (R2020a)      License 68666
% Simulink                                              Version 10.1        (R2020a)      License 68666
% Bioinformatics Toolbox                                Version 4.14        (R2020a)      License 68666
% Computer Vision Toolbox                               Version 9.2         (R2020a)      License 68666
% Curve Fitting Toolbox                                 Version 3.5.11      (R2020a)      License 68666
% Deep Learning Toolbox                                 Version 14.0        (R2020a)      License 68666
% Image Acquisition Toolbox                             Version 6.2         (R2020a)      License 68666
% Image Processing Toolbox                              Version 11.1        (R2020a)      License 68666
% MATLAB Compiler                                       Version 8.0         (R2020a)      License 68666
% MATLAB Compiler SDK                                   Version 6.8         (R2020a)      License 68666
% Optimization Toolbox                                  Version 8.5         (R2020a)      License 68666
% Parallel Computing Toolbox                            Version 7.2         (R2020a)      License 68666
% Signal Processing Toolbox                             Version 8.4         (R2020a)      License 68666
% Statistics and Machine Learning Toolbox               Version 11.7        (R2020a)      License 68666
% Symbolic Math Toolbox                                 Version 8.5         (R2020a)      License 68666
% Wavelet Toolbox                                       Version 5.4         (R2020a)      License 68666
% 
% references:
%    https://www.mathworks.com/matlabcentral/answers/427468-how-does-semanticseg-command-work-on-images-larger-than-what-the-network-was-trained-with
%    https://www.mathworks.com/help/deeplearning/ref/activations.html
clear all 
close all

% load nifti functions
addpath nifti

%% Download Pretrained Network and Sample Test Set
% Optionally, download a pretrained version of 3-D U-Net and five sample test 
% volumes and their corresponding labels from the BraTS data set [3]. The pretrained 
% model and sample data enable you to perform segmentation on test data without 
% downloading the full data set or waiting for the network to train.

trained3DUnet_url = 'https://www.mathworks.com/supportfiles/vision/data/brainTumor3DUNet.mat';
sampleData_url = 'https://www.mathworks.com/supportfiles/vision/data/sampleBraTSTestSet.tar.gz';

imageDir = fullfile(tempdir,'BraTS');
if ~exist(imageDir,'dir')
    mkdir(imageDir);
end
downloadTrained3DUnetSampleData(trained3DUnet_url,sampleData_url,imageDir);

% return a pretrained 3-D U-Net network.
load(fullfile(imageDir,'trained3DUNet','brainTumor3DUNet.mat'));
analyzeNetwork(net)

% You can now use the U-Net to semantically segment brain tumors.
%% Perform Segmentation of Test Data
% load five volumes for testing.
volLocTest = fullfile(imageDir,'sampleBraTSTestSet','imagesTest');
lblLocTest = fullfile(imageDir,'sampleBraTSTestSet','labelsTest');
classNames = ["background","tumor"];
pixelLabelID = [0 1];

%% 
% Crop the central portion of the images and labels to size 128-by-128-by-128 
% voxels by using the helper function |centerCropMatReader|. This function is 
% attached to the example as a supporting file. The |voldsTest| variable stores 
% the ground truth test images. The |pxdsTest| variable stores the ground truth 
% labels.

windowSize = [128 128 128];
volReader = @(x) centerCropMatReader(x,windowSize);
labelReader = @(x) centerCropMatReader(x,windowSize);
voldsTest = imageDatastore(volLocTest, ...
    'FileExtensions','.mat','ReadFcn',volReader);
pxdsTest = pixelLabelDatastore(lblLocTest,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',labelReader);
%% 
% For each test image, add the ground truth image volumes and labels to cell 
% arrays. Use the trained network with the <docid:vision_ref#mw_bbecb1af-a6c9-43d1-91f5-48607edc15d1 
% |semanticseg|> function to predict the labels for each test volume.
% 
% After performing the segmentation, postprocess the predicted labels by labeling 
% nonbrain voxels as |1|, corresponding to the background. Use the test images 
% to determine which voxels do not belong to the brain. You can also clean up 
% the predicted labels by removing islands and filling holes using the <docid:images_ref#bvb_85o-1 
% |medfilt3|> function. |medfilt3| does not support categorical data, so cast 
% the pixel label IDs to |uint8| before the calculation. Then, cast the filtered 
% labels back to the categorical data type, specifying the original pixel label 
% IDs and class names.

id=1;
while hasdata(voldsTest)
    disp(['Processing test volume ' num2str(id)])
    
    tempGroundTruth = read(pxdsTest);
    groundTruthLabels{id} =  tempGroundTruth{1};
    
    vol{id} = read(voldsTest);
    tempSeg = semanticseg(vol{id},net);

    % Get the non-brain region mask from the test image.
    volMask = vol{id}(:,:,:,1)==0;
    % Set the non-brain region of the predicted label as background.
    tempSeg(volMask) = classNames(1);
    % Perform median filtering on the predicted label.
    tempSeg = medfilt3(uint8(tempSeg)-1);
    % Cast the filtered label to categorial.
    tempSeg = categorical(tempSeg,pixelLabelID,classNames);
    predictedLabels{id} = tempSeg;
    id=id+1;
end
%% Compare Ground Truth Against Network Prediction
% Select one of the test images to evaluate the accuracy of the semantic segmentation. 
% Extract the first modality from the 4-D volumetric data and store this 3-D volume 
% in the variable |vol3d|.

volId = 2;
vol3d = vol{volId}(:,:,:,1);
%% 
% Display in a montage the center slice of the ground truth and predicted labels 
% along the depth direction.

zID = size(vol3d,3)/2;
zSliceGT = labeloverlay(vol3d(:,:,zID),groundTruthLabels{volId}(:,:,zID));
zSlicePred = labeloverlay(vol3d(:,:,zID),predictedLabels{volId}(:,:,zID));

figure
title('Labeled Ground Truth (Left) vs. Network Prediction (Right)')
montage({zSliceGT;zSlicePred},'Size',[1 2],'BorderSize',5) 
%% 
% Display the ground-truth labeled volume using the <docid:images_ref#mw_7a40592d-db0e-4bdb-9ba3-446bd1715151 
% |labelvolshow|> function. Make the background fully transparent by setting the 
% visibility of the background label (|1|) to |0|. Because the tumor is inside 
% the brain tissue, make some of the brain voxels transparent, so that the tumor 
% is visible. To make some brain voxels transparent, specify the volume threshold 
% as a number in the range [0, 1]. All normalized volume intensities below this 
% threshold value are fully transparent. This example sets the volume threshold 
% as less than 1 so that some brain pixels remain visible, to give context to 
% the spatial location of the tumor inside the brain.

figure
h1 = labelvolshow(groundTruthLabels{volId},vol3d);
h1.LabelVisibility(1) = 0;
h1.VolumeThreshold = 0.68;
%% 
% For the same volume, display the predicted labels.

figure
h2 = labelvolshow(predictedLabels{volId},vol3d);
h2.LabelVisibility(1) = 0;
h2.VolumeThreshold = 0.68;
%% 
% This image shows the result of displaying slices sequentially across the entire 
% volume.
% 
% %% Quantify Segmentation Accuracy
% Measure the segmentation accuracy using the <docid:images_ref#mw_1ee709d7-bf6b-4ac9-8f5d-e7caf72497d4 
% |dice|> function. This function computes the Dice similarity coefficient between 
% the predicted and ground truth segmentations.

diceResult = zeros(length(voldsTest.Files),2);

for j = 1:length(vol)
    diceResult(j,:) = dice(groundTruthLabels{j},predictedLabels{j});
end
%% 
% Calculate the average Dice score across the set of test volumes.

meanDiceBackground = mean(diceResult(:,1));
disp(['Average Dice score of background across ',num2str(j), ...
    ' test volumes = ',num2str(meanDiceBackground)])
meanDiceTumor = mean(diceResult(:,2));
disp(['Average Dice score of tumor across ',num2str(j), ...
    ' test volumes = ',num2str(meanDiceTumor)])
%% 
% The figure shows a <docid:stats_ug#bu180jd |boxplot|> that visualizes statistics 
% about the Dice scores across the set of five sample test volumes. The red lines 
% in the plot show the median Dice value for the classes. The upper and lower 
% bounds of the blue box indicate the 25th and 75th percentiles, respectively. 
% Black whiskers extend to the most extreme data points not considered outliers.
% 
% % 
% If you have Statistics and Machine Learning Toolbox™, then you can use the 
% |boxplot| function to visualize statistics about the Dice scores across all 
% your test volumes. To create a |boxplot|, set the |createBoxplot| parameter 
% in the following code to |true|.

createBoxplot = true;
if createBoxplot
    figure
    boxplot(diceResult)
    title('Test Set Dice Accuracy')
    xticklabels(classNames)
    ylabel('Dice Coefficient')
end
%% Summary
% This example shows how to create and train a 3-D U-Net network to perform 
% 3-D brain tumor segmentation using the BraTS data set. The steps to train the 
% network include:
%% 
% * Download and preprocess the training data.
% * Create a <docid:images_ref#mw_19a16ac8-a068-411c-8f32-def517a4399a |randomPatchExtractionDatastore|> 
% that feeds training data to the network. 
% * Define the layers of the 3-D U-Net network.
% * Specify training options.
% * Train the network using the <docid:nnet_ref#bu6sn4c |trainNetwork|> function.
%% 
% After training the 3-D U-Net network or loading a pretrained 3-D U-Net network, 
% the example performs semantic segmentation of a test data set. The example evaluates 
% the predicted segmentation by a visual comparison to the ground truth segmentation 
% and by measuring the Dice similarity coefficient between the predicted and ground 
% truth segmentation.
%% References
% [1] Çiçek, Ö., A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger. 
% "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation." In 
% _Proceedings of the International Conference on Medical Image Computing and 
% Computer-Assisted Intervention_. Athens, Greece, Oct. 2016, pp. 424-432.
% 
% [2] Isensee, F., P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein. 
% "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to 
% the BRATS 2017 Challenge." In _Proceedings of BrainLes: International MICCAI 
% Brainlesion Workshop_. Quebec City, Canada, Sept. 2017, pp. 287-297.
% 
% [3] "Brain Tumours". _Medical Segmentation Decathalon._ http://medicaldecathlon.com/ 
% 
% The BraTS dataset is provided by Medical Decathlon under the <https://creativecommons.org/licenses/by-sa/4.0/ 
% CC-BY-SA 4.0 license.> All warranties and representations are disclaimed; see 
% the license for details. MathWorks® has modified the data set linked in the 
% _Download Pretrained Network and Sample Test Set_ section of this example. The 
% modified sample dataset has been cropped to a region containing primarily the 
% brain and tumor and each channel has been normalized independently by subtracting 
% the mean and dividing by the standard deviation of the cropped brain region.
% 
% [4] Sudre, C. H., W. Li, T. Vercauteren, S. Ourselin, and M. J. Cardoso. "Generalised 
% Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations." 
% _Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical 
% Decision Support: Third International Workshop_. Quebec City, Canada, Sept. 
% 2017, pp. 240-248.
% 
% _Copyright 2018 The MathWorks, Inc._


%% extract features from hidden layers
%% https://www.mathworks.com/help/deeplearning/examples/extract-image-features-using-pretrained-network.html

% sanity check
layername  = 'input'     
originalFeatures = activations(net,vol{volId},layername );
norm(originalFeatures(:)-vol{volId}(:))
orignii = make_nii(originalFeatures,[],[],[],'original');
save_nii(orignii,'original.nii' ) ;

% features after first convolution
layername = 'conv_Module2_Level1';
convM2L1 = activations(net,vol{volId},layername );
convnii = make_nii(convM2L1 ,[],[],[],'convolution');
save_nii(convnii,'convolution.nii' ) ;

% features after first batch normalization
layername = 'BN_Module2_Level1'
bnFeaturesM2L1 = activations(net,vol{volId},layername );

% relu features 
layername = 'relu_Module2_Level1'  
reluM2L1 = activations(net,vol{volId},layername );
layername = 'relu_Module2_Level2'  
reluM2L2 = activations(net,vol{volId},layername );

% max pool 
layername = 'maxpool_Module2'      
maxpoolM2 = activations(net,vol{volId},layername );

% input/output to transpose convolution layer
layername  = 'relu_Module6_Level2'
inputFeatures = activations(net,vol{volId},layername );
size(inputFeatures )

layername  = 'transConv_Module6'     
outputFeatures = activations(net,vol{volId},layername );
size(outputFeatures )

% concatenate features
layername  = 'concat1'              
concatFeatures = activations(net,vol{volId},layername );

% last relu
layername  = 'relu_Module7_Level2'  
lastReluFeatures = activations(net,vol{volId},layername);
lastrelunii = make_nii(lastReluFeatures ,[],[],[],'lastrelu');
save_nii(lastrelunii,'lastrelu.nii' ) ;

% last convolution features
size(net.Layers(39).Weights)
% NOTE - no neighborhood covolution. output is a linear combination of the input channels
% ans = 1     1     1    64     2
layername  = 'ConvLast_Module7'
lastconvFeatures = activations(net,vol{volId},layername );
lastconvnii = make_nii(lastconvFeatures,[],[],[],'lastconv');
save_nii(lastconvnii,'lastconv.nii' ) ;

% softmax features
layername  = 'softmax'              
softmaxFeatures = activations(net,vol{volId},layername );
softnii = make_nii(softmaxFeatures,[],[],[],'softmax');
save_nii(softnii,'softmax.nii' ) ;

% output predictions
layername  = 'output'              
testout = activations(net,vol{volId},layername );
norm(softmaxFeatures(:)-testout(:))
[maxvalue , testlabel ]= max(softmaxFeatures,[],4);
% Get the non-brain region mask from the test image.
volMask = vol{volId}(:,:,:,1)==0;
% Set the non-brain region of the predicted label as background.
testlabel(volMask) = classNames(1);
% Perform median filtering on the predicted label.
testlabel = medfilt3(uint8(testlabel)-1);
% Cast the filtered label to categorial.
testlabel = categorical(testlabel,pixelLabelID,classNames);
norm(double(uint8(testlabel(:)) - uint8(predictedLabels{volId}(:))))

% write predictions
segnii = make_nii(uint8(predictedLabels{volId}),[],[],[],'segmentation');
save_nii(segnii,'output.nii' ) ;

% view output
% vglrun itksnap -g original.nii -s output.nii -o convolution.nii lastrelu.nii lastconv.nii softmax.nii 


% setup homework exercise
beta = 2 % find non-zero channel
y1   = convM2L1(:,:,:,beta);
y2   = bnFeaturesM2L1(:,:,:,beta); 
y3   = reluM2L1(:,:,:,beta); 
trainedVariance   = net.Layers(8).TrainedVariance(:,:,:,beta) ;
trainedMean       = net.Layers(8).TrainedMean(:,:,:,beta);
trainedScale      = net.Layers(8).Scale(:,:,:,beta);
trainedOffset     = net.Layers(8).Offset(:,:,:,beta);
betamp = 55 % find non-zero channel 
x1   = reluM2L2(:,:,:,betamp); 
m2   = maxpoolM2(:,:,:,betamp); 
save('nnWorkspace.mat','y1', 'y2', 'y3', 'x1','m2','trainedMean', 'trainedVariance','trainedScale', 'trainedOffset')

