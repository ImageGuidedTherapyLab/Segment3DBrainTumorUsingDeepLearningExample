%% 3-D Brain Tumor Segmentation Using Deep Learning
% This example shows how to train a 3-D U-Net neural network and perform semantic 
% segmentation of brain tumors from 3-D medical images. The example shows how 
% to train a 3-D U-Net network and also provides a pretrained network. Use of 
% a CUDA-capable NVIDIA™ GPU with compute capability 3.0 or higher is highly recommended 
% for 3-D semantic segmentation (requires Parallel Computing Toolbox™).
%% Introduction
% Semantic segmentation involves labeling each pixel in an image or voxel of 
% a 3-D volume with a class. This example illustrates the use of deep learning 
% methods to perform binary semantic segmentation of brain tumors in magnetic 
% resonance imaging (MRI) scans. In this binary segmentation, each pixel is labeled 
% as tumor or background.
% 
% This example performs brain tumor segmentation using a 3-D U-Net architecture 
% [<internal:D0EBDA01 1>]. U-Net is a fast, efficient and simple network that 
% has become popular in the semantic segmentation domain.
% 
% One challenge of medical image segmentation is the amount of memory needed 
% to store and process 3-D volumes. Training a network on the full input volume 
% is impractical due to GPU resource constraints. This example solves the problem 
% by training the network on image patches. The example uses an overlap-tile strategy 
% to stitch test patches into a complete segmented test volume. The example avoids 
% border artifacts by using the valid part of the convolution in the neural network 
% [<internal:3D38B94F 5>].
% 
% A second challenge of medical image segmentation is class imbalance in the 
% data that hampers training when using conventional cross entropy loss. This 
% example solves the problem by using a weighted multiclass Dice loss function 
% [<internal:1585DE3E 4>]. Weighting the classes helps to counter the influence 
% of larger regions on the Dice score, making it easier for the network to learn 
% how to segment smaller regions.
%% Download Training, Validation, and Test Data
% This example uses the BraTS data set [<internal:AF124EDA 2>]. The BraTS data 
% set contains MRI scans of brain tumors, namely gliomas, which are the most common 
% primary brain malignancies. The size of the data file is ~7 GB. If you do not 
% want to download the BraTS data set, then go directly to the <internal:H_594338B4 
% Download Pretrained Network and Sample Test Set> section in this example.
% 
% Create a directory to store the BraTS data set.

imageDir = fullfile(tempdir,'BraTS');
if ~exist(imageDir,'dir')
    mkdir(imageDir);
end
%% 
% To download the BraTS data, go to the <http://medicaldecathlon.com/ Medical 
% Segmentation Decathlon> website and click the "Download Data" link. Download 
% the "Task01_BrainTumour.tar" file [<internal:AAAA5FC0 3>]. Unzip the TAR file 
% into the directory specified by the |imageDir| variable. When unzipped successfully, 
% |imageDir| will contain a directory named |Task01_BrainTumour| that has three 
% subdirectories: |imagesTr|, |imagesTs|, and |labelsTr|. 
% 
% The data set contains 750 4-D volumes, each representing a stack of 3-D images. 
% Each 4-D volume has size 240-by-240-by-155-by-4, where the first three dimensions 
% correspond to height, width, and depth of a 3-D volumetric image. The fourth 
% dimension corresponds to different scan modalities. The data set is divided 
% into 484 training volumes with voxel labels and 266 test volumes, The test volumes 
% do not have labels so this example does not use the test data. Instead, the 
% example splits the 484 training volumes into three independent sets that are 
% used for training, validation, and testing.
%% Preprocess Training and Validation Data
% To train the 3-D U-Net network more efficiently, preprocess the MRI data using 
% the helper function |preprocessBraTSdataset|. This function is attached to the 
% example as a supporting file.
% 
% The helper function performs these operations:
%% 
% * Crop the data to a region containing primarily the brain and tumor. Cropping 
% the data reduces the size of data while retaining the most critical part of 
% each MRI volume and its corresponding labels.
% * Normalize each modality of each volume independently by subtracting the 
% mean and dividing by the standard deviation of the cropped brain region.
% * Split the 484 training volumes into 400 training, 29 validation, and 55 
% test sets. 
%% 
% Preprocessing the data can take about 30 minutes to complete. 

sourceDataLoc = [imageDir filesep 'Task01_BrainTumour'];
preprocessDataLoc = fullfile(tempdir,'BraTS','preprocessedDataset');
preprocessBraTSdataset(preprocessDataLoc,sourceDataLoc);
%% Create Random Patch Extraction Datastore for Training and Validation
% Use a random patch extraction datastore to feed the training data to the network 
% and to validate the training progress. This datastore extracts random patches 
% from ground truth images and corresponding pixel label data. Patching is a common 
% technique to prevent running out of memory when training with arbitrarily large 
% volumes.
% 
% Create an <docid:matlab_ref#butueui-1 |imageDatastore|> to store the 3-D image 
% data. Because the MAT-file format is a nonstandard image format, you must use 
% a MAT-file reader to enable reading the image data. You can use the helper MAT-file 
% reader, |matRead|. This function is attached to the example as a supporting 
% file. 

volReader = @(x) matRead(x);
volLoc = fullfile(preprocessDataLoc,'imagesTr');
volds = imageDatastore(volLoc, ...
    'FileExtensions','.mat','ReadFcn',volReader);
%% 
% Create a <docid:vision_ref#mw_c2246553-ba4a-4bad-aad4-6ab8fa2f7f2d |pixelLabelDatastore|> 
% to store the labels.

lblLoc = fullfile(preprocessDataLoc,'labelsTr');
classNames = ["background","tumor"];
pixelLabelID = [0 1];
pxds = pixelLabelDatastore(lblLoc,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',volReader);
%% 
% Preview one image volume and label. Display the labeled volume using the <docid:images_ref#mw_7a40592d-db0e-4bdb-9ba3-446bd1715151 
% |labelvolshow|> function. Make the background fully transparent by setting the 
% visibility of the background label (|1|) to |0|.

volume = preview(volds);
label = preview(pxds);

viewPnl = uipanel(figure,'Title','Labeled Training Volume');
hPred = labelvolshow(label,volume(:,:,:,1),'Parent',viewPnl, ...
    'LabelColor',[0 0 0;1 0 0]);
hPred.LabelVisibility(1) = 0;
%% 
% Create a <docid:images_ref#mw_19a16ac8-a068-411c-8f32-def517a4399a |randomPatchExtractionDatastore|> 
% that contains the training image and pixel label data. Specify a patch size 
% of 132-by-132-by-132 voxels. Specify |'PatchesPerImage'| to extract 16 randomly 
% positioned patches from each pair of volumes and labels during training. Specify 
% a mini-batch size of 8.

patchSize = [132 132 132];
patchPerImage = 16;
miniBatchSize = 8;
patchds = randomPatchExtractionDatastore(volds,pxds,patchSize, ...
    'PatchesPerImage',patchPerImage);
patchds.MiniBatchSize = miniBatchSize;
%% 
% Follow the same steps to create a |randomPatchExtractionDatastore| that contains 
% the validation image and pixel label data. You can use validation data to evaluate 
% whether the network is continuously learning, underfitting, or overfitting as 
% time progresses.

volLocVal = fullfile(preprocessDataLoc,'imagesVal');
voldsVal = imageDatastore(volLocVal, ...
    'FileExtensions','.mat','ReadFcn',volReader);

lblLocVal = fullfile(preprocessDataLoc,'labelsVal');
pxdsVal = pixelLabelDatastore(lblLocVal,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',volReader);

dsVal = randomPatchExtractionDatastore(voldsVal,pxdsVal,patchSize, ...
    'PatchesPerImage',patchPerImage);
dsVal.MiniBatchSize = miniBatchSize;
%% 
% Augment the training and validation data by using the <docid:matlab_ref#mw_16489124-fe7e-4381-b715-8d3b8b30a9f6 
% |transform|> function with custom preprocessing operations specified by the 
% helper function |augmentAndCrop3dPatch|. This function is attached to the example 
% as a supporting file.
% 
% The |augmentAndCrop3dPatch| function performs these operations:
%% 
% # Randomly rotate and reflect training data to make the training more robust. 
% The function does not rotate or reflect validation data.
% # Crop response patches to the output size of the network, 44-by-44-by-44 
% voxels.

dataSource = 'Training';
dsTrain = transform(patchds,@(patchIn)augmentAndCrop3dPatch(patchIn,dataSource));

dataSource = 'Validation';
dsVal = transform(dsVal,@(patchIn)augmentAndCrop3dPatch(patchIn,dataSource));
%% Set Up 3-D U-Net Layers
% This example uses the 3-D U-Net network [<internal:D0EBDA01 1>]. In U-Net, 
% the initial series of convolutional layers are interspersed with max pooling 
% layers, successively decreasing the resolution of the input image. These layers 
% are followed by a series of convolutional layers interspersed with upsampling 
% operators, successively increasing the resolution of the input image. A batch 
% normalization layer is introduced before each ReLU layer. The name U-Net comes 
% from the fact that the network can be drawn with a symmetric shape like the 
% letter U.
% 
% Create a default 3-D U-Net network by using the <docid:vision_ref#mw_a0372434-cb34-4b9f-8dad-97ab7de4119b 
% |unet3dLayers|> function. Specify two class segmentation. Also specify valid 
% convolution padding to avoid border artifacts when using the overlap-tile strategy 
% for prediction of the test volumes.

inputPatchSize = [132 132 132 4];
numClasses = 2;
[lgraph,outPatchSize] = unet3dLayers(inputPatchSize,numClasses,'ConvolutionPadding','valid');
%% 
% To better segment smaller tumor regions and reduce the influence of larger 
% background regions, this example uses a <docid:vision_ref#mw_d42d8f29-43c0-47ab-b548-dd56365ad3bb 
% |dicePixelClassificationLayer|>. Replace the pixel classification layer with 
% the Dice pixel classification layer.

outputLayer = dicePixelClassificationLayer('Name','Output');
lgraph = replaceLayer(lgraph,'Segmentation-Layer',outputLayer);
%% 
% The data has already been normalized in the <internal:H_E5B3A73E Preprocess 
% Training and Validation Data> section of this example. Data normalization in 
% the <docid:nnet_ref#object_layer.imageinput3dlayer |imageInput3dLayer|> is unnecessary, 
% so replace the input layer with an input layer that does not have data normalization.

inputLayer = image3dInputLayer(inputPatchSize,'Normalization','none','Name','ImageInputLayer');
lgraph = replaceLayer(lgraph,'ImageInputLayer',inputLayer);
%% 
% Alternatively, you can modify the 3-D U-Net network by using Deep Network 
% Designer App from Deep Learning Toolbox™.
% 
% Plot the graph of the updated 3-D U-Net network.

analyzeNetwork(lgraph)
%% Specify Training Options
% Train the network using the |adam| optimization solver. Specify the hyperparameter 
% settings using the <docid:nnet_ref#bu59f0q |trainingOptions|> function. The 
% initial learning rate is set to 5e-4 and gradually decreases over the span of 
% training. You can experiment with the |MiniBatchSize| property based on your 
% GPU memory. To maximize GPU memory utilization, favor large input patches over 
% a large batch size. Note that batch normalization layers are less effective 
% for smaller values of |MiniBatchSize|. Tune the initial learning rate based 
% on the |MiniBatchSize|.

options = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'InitialLearnRate',5e-4, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',5, ...
    'LearnRateDropFactor',0.95, ...
    'ValidationData',dsVal, ...
    'ValidationFrequency',400, ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'MiniBatchSize',miniBatchSize);
%% Download Pretrained Network and Sample Test Set
% Optionally, download a pretrained version of 3-D U-Net and five sample test 
% volumes and their corresponding labels from the BraTS data set [<internal:AAAA5FC0 
% 3>]. The pretrained model and sample data enable you to perform segmentation 
% on test data without downloading the full data set or waiting for the network 
% to train.

trained3DUnet_url = 'https://www.mathworks.com/supportfiles/vision/data/brainTumor3DUNetValid.mat';
sampleData_url = 'https://www.mathworks.com/supportfiles/vision/data/sampleBraTSTestSetValid.tar.gz';

imageDir = fullfile(tempdir,'BraTS');
if ~exist(imageDir,'dir')
    mkdir(imageDir);
end

downloadTrained3DUnetSampleData(trained3DUnet_url,sampleData_url,imageDir);
%% Train Network
% After configuring the training options and the data source, train the 3-D 
% U-Net network by using the <docid:nnet_ref#bu6sn4c |trainNetwork|> function. 
% To train the network, set the |doTraining| variable in the following code to 
% |true|. A CUDA capable NVIDIA™ GPU with compute capability 3.0 or higher is 
% highly recommended for training.
% 
% If you keep the |doTraining| variable in the following code as |false|, then 
% the example returns a pretrained 3-D U-Net network.
% 
% _Note: Training takes about 30 hours on a multi-GPU system with 4 NVIDIA™ 
% Titan Xp GPUs and can take even longer depending on your GPU hardware._

doTraining = false;
if doTraining
    modelDateTime = datestr(now,'dd-mmm-yyyy-HH-MM-SS');
    [net,info] = trainNetwork(dsTrain,lgraph,options);
    save(['trained3DUNetValid-' modelDateTime '-Epoch-' num2str(options.MaxEpochs) '.mat'],'net');
else
    inputPatchSize = [132 132 132 4];
    outPatchSize = [44 44 44 2];
    load(fullfile(imageDir,'trained3DUNet','brainTumor3DUNetValid.mat'));
end
%% 
% You can now use the U-Net to semantically segment brain tumors.
%% Perform Segmentation of Test Data
% A GPU is highly recommended for performing semantic segmentation of the image 
% volumes (requires Parallel Computing Toolbox™).
% 
% Select the source of test data that contains ground truth volumes and labels 
% for testing. If you keep the |useFullTestSet| variable in the following code 
% as |false|, then the example uses five volumes for testing. If you set the |useFullTestSet| 
% variable to |true|, then the example uses 55 test images selected from the full 
% data set.

useFullTestSet = false;
if useFullTestSet
    volLocTest = fullfile(preprocessDataLoc,'imagesTest');
    lblLocTest = fullfile(preprocessDataLoc,'labelsTest');
else
    volLocTest = fullfile(imageDir,'sampleBraTSTestSetValid','imagesTest');
    lblLocTest = fullfile(imageDir,'sampleBraTSTestSetValid','labelsTest');
    classNames = ["background","tumor"];
    pixelLabelID = [0 1];
end
%% 
% The |voldsTest| variable stores the ground truth test images. The |pxdsTest| 
% variable stores the ground truth labels.

volReader = @(x) matRead(x);
voldsTest = imageDatastore(volLocTest, ...
    'FileExtensions','.mat','ReadFcn',volReader);
pxdsTest = pixelLabelDatastore(lblLocTest,classNames,pixelLabelID, ...
    'FileExtensions','.mat','ReadFcn',volReader);
%% 
% Use the overlap-tile strategy to predict the labels for each test volume. 
% Each test volume is padded to make the input size a multiple of the output size 
% of the network and compensates for the effects of valid convolution. The overlap-tile 
% algorithm selects overlapping patches, predicts the labels for each patch by 
% using the <docid:vision_ref#mw_bbecb1af-a6c9-43d1-91f5-48607edc15d1 |semanticseg|> 
% function, and then recombines the patches.

id = 1;
while hasdata(voldsTest)
    disp(['Processing test volume ' num2str(id)]);
    
    tempGroundTruth = read(pxdsTest);
    groundTruthLabels{id} = tempGroundTruth{1};
    vol{id} = read(voldsTest);
    
    % Use reflection padding for the test image. 
    % Avoid padding of different modalities.
    volSize = size(vol{id},(1:3));
    padSizePre  = (inputPatchSize(1:3)-outPatchSize(1:3))/2;
    padSizePost = (inputPatchSize(1:3)-outPatchSize(1:3))/2 + (outPatchSize(1:3)-mod(volSize,outPatchSize(1:3)));
    volPaddedPre = padarray(vol{id},padSizePre,'symmetric','pre');
    volPadded = padarray(volPaddedPre,padSizePost,'symmetric','post');
    [heightPad,widthPad,depthPad,~] = size(volPadded);
    [height,width,depth,~] = size(vol{id});
    
    tempSeg = categorical(zeros([height,width,depth],'uint8'),[0;1],classNames);
    
    % Overlap-tile strategy for segmentation of volumes.
    for k = 1:outPatchSize(3):depthPad-inputPatchSize(3)+1
        for j = 1:outPatchSize(2):widthPad-inputPatchSize(2)+1
            for i = 1:outPatchSize(1):heightPad-inputPatchSize(1)+1
                patch = volPadded( i:i+inputPatchSize(1)-1,...
                    j:j+inputPatchSize(2)-1,...
                    k:k+inputPatchSize(3)-1,:);
                patchSeg = semanticseg(patch,net);
                tempSeg(i:i+outPatchSize(1)-1, ...
                    j:j+outPatchSize(2)-1, ...
                    k:k+outPatchSize(3)-1) = patchSeg;
            end
        end
    end
    
    % Crop out the extra padded region.
    tempSeg = tempSeg(1:height,1:width,1:depth);

    % Save the predicted volume result.
    predictedLabels{id} = tempSeg;
    id=id+1;
end
%% Compare Ground Truth Against Network Prediction
% Select one of the test images to evaluate the accuracy of the semantic segmentation. 
% Extract the first modality from the 4-D volumetric data and store this 3-D volume 
% in the variable |vol3d|.

volId = 1;
vol3d = vol{volId}(:,:,:,1);
%% 
% Display in a montage the center slice of the ground truth and predicted labels 
% along the depth direction.

zID = size(vol3d,3)/2;
zSliceGT = labeloverlay(vol3d(:,:,zID),groundTruthLabels{volId}(:,:,zID));
zSlicePred = labeloverlay(vol3d(:,:,zID),predictedLabels{volId}(:,:,zID));

figure
montage({zSliceGT,zSlicePred},'Size',[1 2],'BorderSize',5) 
title('Labeled Ground Truth (Left) vs. Network Prediction (Right)')
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

viewPnlTruth = uipanel(figure,'Title','Ground-Truth Labeled Volume');
hTruth = labelvolshow(groundTruthLabels{volId},vol3d,'Parent',viewPnlTruth, ...
    'LabelColor',[0 0 0;1 0 0],'VolumeThreshold',0.68);
hTruth.LabelVisibility(1) = 0;
%% 
% For the same volume, display the predicted labels.

viewPnlPred = uipanel(figure,'Title','Predicted Labeled Volume');
hPred = labelvolshow(predictedLabels{volId},vol3d,'Parent',viewPnlPred, ...
    'LabelColor',[0 0 0;1 0 0],'VolumeThreshold',0.68);
hPred.LabelVisibility(1) = 0;
%% 
% This image shows the result of displaying slices sequentially across the one 
% of the volume. The labeled ground truth is on the left and the network prediction 
% is on the right.
% 
% 
%% Quantify Segmentation Accuracy
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
% 
% 
% If you have Statistics and Machine Learning Toolbox™, then you can use the 
% |boxplot| function to visualize statistics about the Dice scores across all 
% your test volumes. To create a |boxplot|, set the |createBoxplot| variable in 
% the following code to |true|.

createBoxplot = false;
if createBoxplot
    figure
    boxplot(diceResult)
    title('Test Set Dice Accuracy')
    xticklabels(classNames)
    ylabel('Dice Coefficient')
end
%% References
% [1] Çiçek, Ö., A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger. 
% "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation." In 
% _Proceedings of the International Conference on Medical Image Computing and 
% Computer-Assisted Intervention - MICCAI 2016_. Athens, Greece, Oct. 2016, pp. 
% 424-432.
% 
% [2] Isensee, F., P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein. 
% "Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to 
% the BRATS 2017 Challenge." In _Proceedings of BrainLes: International MICCAI 
% Brainlesion Workshop_. Quebec City, Canada, Sept. 2017, pp. 287-297.
% 
% [3] "Brain Tumours". _Medical Segmentation Decathlon._ http://medicaldecathlon.com/ 
% 
% The BraTS dataset is provided by Medical Segmentation Decathlon under the 
% <https://creativecommons.org/licenses/by-sa/4.0/ CC-BY-SA 4.0 license.> All 
% warranties and representations are disclaimed; see the license for details. 
% MathWorks® has modified the data set linked in the <internal:H_594338B4 Download 
% Pretrained Network and Sample Test Set> section of this example. The modified 
% sample dataset has been cropped to a region containing primarily the brain and 
% tumor and each channel has been normalized independently by subtracting the 
% mean and dividing by the standard deviation of the cropped brain region.
% 
% [4] Sudre, C. H., W. Li, T. Vercauteren, S. Ourselin, and M. J. Cardoso. "Generalised 
% Dice Overlap as a Deep Learning Loss Function for Highly Unbalanced Segmentations." 
% _Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical 
% Decision Support: Third International Workshop_. Quebec City, Canada, Sept. 
% 2017, pp. 240-248.
% 
% [5] Ronneberger, O., P. Fischer, and T. Brox. "U-Net:Convolutional Networks 
% for Biomedical Image Segmentation." In _Proceedings of the International Conference 
% on Medical Image Computing and Computer-Assisted Intervention - MICCAI 2015_. 
% Munich, Germany, Oct. 2015, pp. 234-241. Available at arXiv:1505.04597.
% 
% _Copyright 2019 The MathWorks, Inc._