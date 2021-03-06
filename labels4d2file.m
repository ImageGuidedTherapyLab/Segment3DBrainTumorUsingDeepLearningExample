function labels4d2file(imVol,folderPath, instanceNumber)

%This function slices 4d images through the axial point (3rd dimension)

% find number of slices
slices = size(imVol, 3);

% find number of channels
numChannel = size(imVol, 4);

% Resize to 128
Dat = zeros(128,128,slices, numChannel);
for i = 1:slices
    Dat(:,:,i,:) = imresize(imVol(:,:,i,:),[128 128]); 
end

% Write the images to file as .mat
num = 0; if(~isempty(instanceNumber) ), num = instanceNumber; end
for i = 0:slices-1
    cropVol = Dat(:,:,i+1,:);
    save([sprintf('%s/Img_%06d_PT%d.mat',folderPath,i+num, instanceNumber)],'cropVol');
    
end

end
