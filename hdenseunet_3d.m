%% Create Layer Graph
% Create the layer graph variable to contain the network layers.

lgraph = layerGraph();
%% Add Layer Branches
% Add the branches of the network to the layer graph. Each branch is a linear 
% array of layers.

tempLayers = [
    image3dInputLayer([224 224 12 3],"Name","image3dinput")
    convolution3dLayer([7 7 7],96,"Name","conv3d_1","Padding","same","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling3dLayer([3 3 3],"Name","maxpool3d","Padding","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","batchnorm_1_1")
    reluLayer("Name","relu_1_1")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_1")
    reluLayer("Name","relu_2_1")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_2")
    reluLayer("Name","relu_1_2")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_2")
    reluLayer("Name","relu_2_2")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_1")
    reluLayer("Name","relu_1_3_1")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_1")
    reluLayer("Name","relu_2_3_1")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_1","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","batchnorm_1_4_1")
    reluLayer("Name","relu_1_4_1")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_4_1","Padding","same")
    averagePooling3dLayer([2 2 1],"Name","avgpool3d_1","Padding","same","Stride",[2 2 1])
    batchNormalizationLayer("Name","batchnorm_1_3_2")
    reluLayer("Name","relu_1_3_2")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_2")
    reluLayer("Name","relu_2_3_2")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_3")
    reluLayer("Name","relu_1_3_3")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_3")
    reluLayer("Name","relu_2_3_3")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_3","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_4")
    reluLayer("Name","relu_1_3_4")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_4")
    reluLayer("Name","relu_2_3_4")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_4","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_5")
    reluLayer("Name","relu_1_3_5")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_5","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_5")
    reluLayer("Name","relu_2_3_5")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_5","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","batchnorm_1_4_2")
    reluLayer("Name","relu_1_4_2")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_4_2","Padding","same")
    averagePooling3dLayer([2 2 1],"Name","avgpool3d_2","Padding","same","Stride",[2 2 1])
    batchNormalizationLayer("Name","batchnorm_1_3_6")
    reluLayer("Name","relu_1_3_6")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_6")
    reluLayer("Name","relu_2_3_6")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_6","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_7")
    reluLayer("Name","relu_1_3_7")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_7","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_7")
    reluLayer("Name","relu_2_3_7")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_7","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_8")
    reluLayer("Name","relu_1_3_8")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_8")
    reluLayer("Name","relu_2_3_8")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_8","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_9")
    reluLayer("Name","relu_1_3_9")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_9")
    reluLayer("Name","relu_2_3_9")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_9","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_10")
    reluLayer("Name","relu_1_3_10")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_10","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_10")
    reluLayer("Name","relu_2_3_10")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_10","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_11")
    reluLayer("Name","relu_1_3_11")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_11")
    reluLayer("Name","relu_2_3_11")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_11","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_12")
    reluLayer("Name","relu_1_3_12")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_12","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_12")
    reluLayer("Name","relu_2_3_12")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_12","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_13")
    reluLayer("Name","relu_1_3_13")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_13","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_13")
    reluLayer("Name","relu_2_3_13")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_13","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_14")
    reluLayer("Name","relu_1_3_14")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_14","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_14")
    reluLayer("Name","relu_2_3_14")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_14","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_15")
    reluLayer("Name","relu_1_3_15")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_15","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_15")
    reluLayer("Name","relu_2_3_15")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_15","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_16")
    reluLayer("Name","relu_1_3_16")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_16","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_16")
    reluLayer("Name","relu_2_3_16")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_16","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_17")
    reluLayer("Name","relu_1_3_17")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_17","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_17")
    reluLayer("Name","relu_2_3_17")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_17","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","batchnorm_1_4_3")
    reluLayer("Name","relu_1_4_3")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_4_3","Padding","same")
    averagePooling3dLayer([2 2 1],"Name","avgpool3d_3","Padding","same","Stride",[2 2 1])
    batchNormalizationLayer("Name","batchnorm_1_3_18")
    reluLayer("Name","relu_1_3_18")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_18","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_18")
    reluLayer("Name","relu_2_3_18")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_18","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_19")
    reluLayer("Name","relu_1_3_19")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_19","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_19")
    reluLayer("Name","relu_2_3_19")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_19","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_20")
    reluLayer("Name","relu_1_3_20")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_20","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_20")
    reluLayer("Name","relu_2_3_20")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_20","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_21")
    reluLayer("Name","relu_1_3_21")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_21","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_21")
    reluLayer("Name","relu_2_3_21")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_21","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_22")
    reluLayer("Name","relu_1_3_22")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_22","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_22")
    reluLayer("Name","relu_2_3_22")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_22","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_23")
    reluLayer("Name","relu_1_3_23")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_23","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_23")
    reluLayer("Name","relu_2_3_23")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_23","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_24")
    reluLayer("Name","relu_1_3_24")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_24","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_24")
    reluLayer("Name","relu_2_3_24")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_24","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1_3_25")
    reluLayer("Name","relu_1_3_25")
    convolution3dLayer([1 1 1],128,"Name","conv3d_2_3_25","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2_3_25")
    reluLayer("Name","relu_2_3_25")
    convolution3dLayer([3 3 3],32,"Name","conv3d_3_3_25","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    transposedConv3dLayer([2 2 1],504,"Name","transposed-conv3d_1","Cropping","same","Stride",[2 2 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat_1")
    batchNormalizationLayer("Name","batchnorm_2_3_26")
    reluLayer("Name","relu_2_3_26")
    convolution3dLayer([1 1 1],504,"Name","conv3d_3_3_26","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    transposedConv3dLayer([2 2 1],224,"Name","transposed-conv3d_2","Cropping","same","Stride",[2 2 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat_2")
    batchNormalizationLayer("Name","batchnorm_2_3_27")
    reluLayer("Name","relu_2_3_27")
    convolution3dLayer([1 1 1],224,"Name","conv3d_3_3_27","Padding","same")
    batchNormalizationLayer("Name","batchnorm_3")
    reluLayer("Name","relu_3")
    transposedConv3dLayer([2 2 1],192,"Name","transposed-conv3d_3","Cropping","same","Stride",[2 2 1])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat_3")
    batchNormalizationLayer("Name","batchnorm_2_3_28")
    reluLayer("Name","relu_2_3_28")
    convolution3dLayer([1 1 1],192,"Name","conv3d_3_3_28","Padding","same")
    batchNormalizationLayer("Name","batchnorm_4")
    reluLayer("Name","relu_4")
    transposedConv3dLayer([2 2 2],96,"Name","transposed-conv3d_4","Cropping","same","Stride",[2 2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(4,2,"Name","concat_4")
    batchNormalizationLayer("Name","batchnorm_2_3_29")
    reluLayer("Name","relu_2_3_29")
    convolution3dLayer([1 1 1],96,"Name","conv3d_3_3_29","Padding","same")
    batchNormalizationLayer("Name","batchnorm_5")
    reluLayer("Name","relu_5")
    transposedConv3dLayer([2 2 2],64,"Name","transposed-conv3d_5","Cropping","same","Stride",[2 2 2])
    batchNormalizationLayer("Name","batchnorm_2_3_30")
    reluLayer("Name","relu_2_3_30")
    convolution3dLayer([1 1 1],3,"Name","conv3d_3_3_30","Padding","same")
    softmaxLayer("Name","softmax")
    dicePixelClassificationLayer("Name","dice-pixel-class")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;
%% Connect Layer Branches
% Connect all the branches of the network to create the network graph.

lgraph = connectLayers(lgraph,"conv3d_1","maxpool3d");
lgraph = connectLayers(lgraph,"conv3d_1","concat_4/in2");
lgraph = connectLayers(lgraph,"conv3d_3_3_1","batchnorm_1_4_1");
lgraph = connectLayers(lgraph,"conv3d_3_3_1","concat_3/in2");
lgraph = connectLayers(lgraph,"conv3d_3_3_5","batchnorm_1_4_2");
lgraph = connectLayers(lgraph,"conv3d_3_3_5","concat_2/in2");
lgraph = connectLayers(lgraph,"conv3d_3_3_17","batchnorm_1_4_3");
lgraph = connectLayers(lgraph,"conv3d_3_3_17","concat_1/in1");
lgraph = connectLayers(lgraph,"transposed-conv3d_1","concat_1/in2");
lgraph = connectLayers(lgraph,"transposed-conv3d_2","concat_2/in1");
lgraph = connectLayers(lgraph,"transposed-conv3d_3","concat_3/in1");
lgraph = connectLayers(lgraph,"transposed-conv3d_4","concat_4/in1");
%% Plot Layers

plot(lgraph);
