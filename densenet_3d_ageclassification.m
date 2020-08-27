% Create Layer Graph

lgraph = layerGraph();
%%
n = 4; %this is the number of channels of the input image
numClasses = 2; %this is the number of categories an image could be classified as (in example it was 3, for adult, 7-12, 3-5)
%% Add Layer Branches
% Add the branches of the network to the layer graph. Each branch is a linear 
% array of layers.

tempLayers = [
    image3dInputLayer([64 64 64 4],"Name","input","Normalization","none")
    batchNormalizationLayer("Name","BN_Module1_Level1")
    convolution3dLayer([3 3 3],32,"Name","conv_Module1_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module1_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module1_Level2")
    convolution3dLayer([3 3 3],64,"Name","conv_Module1_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module1_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","depthcat_1")
    maxPooling3dLayer([2 2 2],"Name","maxpool_Module1","Padding","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","BN_Module2_Level1")
    convolution3dLayer([3 3 3],64,"Name","conv_Module2_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module2_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module2_Level2")
    convolution3dLayer([3 3 3],128,"Name","conv_Module2_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module2_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","depthcat_2")
    maxPooling3dLayer([2 2 2],"Name","maxpool_Module2","Padding","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","BN_Module3_Level1")
    convolution3dLayer([3 3 3],128,"Name","conv_Module3_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module3_Level1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","BN_Module3_Level2")
    convolution3dLayer([3 3 3],256,"Name","conv_Module3_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module3_Level2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","depthcat_3")
    maxPooling3dLayer([2 2 2],"Name","maxpool_Module3","Padding","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","BN_Module4_Level1")
    convolution3dLayer([3 3 3],256,"Name","conv_Module4_Level1","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module4_Level1")
    batchNormalizationLayer("Name","BN_Module4_Level2")
    convolution3dLayer([3 3 3],512,"Name","conv_Module4_Level2","Padding","same","WeightsInitializer","narrow-normal")
    reluLayer("Name","relu_Module4_Level2")
    globalAveragePooling3dLayer("Name","pool5")
    fullyConnectedLayer(numClasses,"Name","new_fc","BiasLearnRateFactor",10,"WeightLearnRateFactor",10)
    softmaxLayer("Name","softmax")
    classificationLayer("Name","new_classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;
%% Connect Layer Branches
% Connect all the branches of the network to create the network graph.

lgraph = connectLayers(lgraph,"relu_Module1_Level1","BN_Module1_Level2");
lgraph = connectLayers(lgraph,"relu_Module1_Level1","depthcat_1/in1");
lgraph = connectLayers(lgraph,"relu_Module1_Level2","depthcat_1/in2");
lgraph = connectLayers(lgraph,"relu_Module2_Level1","BN_Module2_Level2");
lgraph = connectLayers(lgraph,"relu_Module2_Level1","depthcat_2/in1");
lgraph = connectLayers(lgraph,"relu_Module2_Level2","depthcat_2/in2");
lgraph = connectLayers(lgraph,"relu_Module3_Level1","BN_Module3_Level2");
lgraph = connectLayers(lgraph,"relu_Module3_Level1","depthcat_3/in1");
lgraph = connectLayers(lgraph,"relu_Module3_Level2","depthcat_3/in2");
%% Plot Layers

plot(lgraph);
