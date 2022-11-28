%% Averaging Filter
% A low pass filter in photos eliminates rainbow patters in photos. A
% lot of extreme details are lost, but the rainbow problem is solved.
clc

img0 = imread('Lena.bmp');
figure(1)
imshow(img0)

img1 = img0(101:200, 101:200);
filt_avg1 = ones(3,3)/9;
img1_LP = uint8(filter2( filt_avg1, img1));

filt_avg2 = ones(3,3)/16; 
filt_avg2(2,2) = 0.5;

img1_LP2 = imfilter( img1, filt_avg2 ); 
figure(2) 
subplot(1, 3, 1), imshow(img1), title('Original') 
subplot(1, 3, 2), imshow(img1_LP), title('Averaging Filter 1') 
subplot(1, 3, 3), imshow(img1_LP2), title('Averaging Filter 2')

%% Sharpen Picture
% The third line of code removes the values outside the 8 bit range and  changes them to uint8.

img1_HP = double(img1) - double(img1_LP); 
figure(3) 
subplot(1, 2, 1), imshow(img1), title('Original'); 
axis equal
subplot(1, 2, 2), imshow(img1_HP, []), title('High-pass Filter1'),  colorbar
SP_factor = 1; % parameter to control the degree of sharpening 

temp = double(img1) + double(img1_HP) * SP_factor; 
img1_SP = uint8( (temp > 255) * 255 + (temp >= 0 & temp <= 255) .*  temp ); 
subplot(1, 3, 3), imshow(img1_SP), title('Sharpened')

%% Edge Filtering
%Edge_filter1 looks at the changes in a horizontal direction, while  Edge_filter2 looks at changes in a vertical direction.

edge_filt1_1 = [-1, 0, 1; -1, 0, 1; -1, 0, 1]; % "Prewitt" edge  filters 
edge_filt1_2 = [-1, -1, -1; 0, 0, 0; 1, 1, 1]; 

img1_edge1_1 = filter2( edge_filt1_1 / 3, img1 ); % apply edge filters
img1_edge1_2 = filter2( edge_filt1_2 / 3, img1 );
img1_edge1_2(1,:) = 0; % handle cases on image border 

img1_edge1_1(:, 1) = 0; 
img1_edge1_2(size(img1,1),:) = 0; 
img1_edge1_1(:, size(img1,2)) = 0;
img1_edge1 = sqrt( double(img1_edge1_1) .^2 +  double(img1_edge1_2) .^2 ); % edge strength 
edge_th = 40; % threshold of edge strength
figure(4)

subplot(2, 3, 1), imshow(img1), title('Original') 
subplot(2, 3, 2), imshow(img1_edge1_1, []), title('After Edge  Filter-1'), colorbar
subplot(2, 3, 3), imshow(img1_edge1_2, []), title('After Edge  Filter-2'), colorbar 
subplot(2, 3, 4), imshow(img1_edge1, []), title('Edge Strength'),  colorbar
subplot(2, 3, 5), imshow(img1_edge1 > edge_th, []), title('Thresholded  Strength ') 
axis equal 
subplot(2, 3, 6), st=1; %% visualize edge directions at a selected resolution 

quiver(img1_edge1_2( size(img1,1):(-st):1, 1:st:size(img1,2)), img1_edge1_1( size(img1,1):(-st):1, 1:st:size(img1,2)));

axis([1, size(img1,2), 1, size(img1,1) ]), axis off, title('Edge Direction (zoom-in to view)')

%% Edge Direction
% The arrows are pointing at different directions at different areas of  the image. 
% For the lips, they seem to be pointing mostly right, for  the eyes, up and down, 
% for the hood, up and down as well, etc.

figure(5)
st = 1;
quiver( img1_edge1_2( size(img1,1):(-st):1, 1:st:size(img1,2)), img1_edge1_1( size(img1,1):(-st):1, 1:st:size(img1,2) ))
axis([1, size(img1,2), 1, size(img1,1) ]), axis off, title('Edge Direction (zoom-in to view)')

edge_filt2_1 = [-1, 0, 1; -2, 0, 2; -1, 0, 1]; % "Sobel" edge filters 
edge_filt2_2 = [-1, -2, -1; 0, 0, 0; 1, 2, 1];
img1_edge2_1 = filter2( edge_filt2_1 / 4, img1 ); % apply edge filters 
img1_edge2_2 = filter2( edge_filt2_2 / 4, img1 );

%% Edge Detector
% The image is first converted into grayscale, and then the circles are 
% found by using the imfindcircles tool. It has parameters that can be 
% added, like ‘ObjectPolarity’ and ‘sensitivity’, which was used for 
% finding dark circles. For the lighter ones, the ‘dark’ parameter is 
% changed to ‘light’, and another parameter named ‘Edge Threshold’ is  added.
% You still must play around with the sensitivity value until  desired result
% is found.

img1_edge3 = edge(img1,'sobel'); 
img1_edge4 = edge(img1,'canny'); 
figure(6) 

subplot(1, 3, 1)
imshow(img1), title('Original') 
subplot(1, 3, 2), imshow(img1_edge3), title('Sobel Edge Detector') 
subplot(1, 3, 3), imshow(img1_edge4), title('Canny Edge Detector')

%% Detecting Circles

figure(7) 
rgb = imread('coloredChips.png'); 
imshow(rgb) 
d = imdistline; 
delete(d) 
gray_image = rgb2gray(rgb); 
imshow(gray_image) 

[centers,radii] = imfindcircles(rgb,[20 25],'ObjectPolarity','dark');
[centers,radii] = imfindcircles(rgb,[20 25],'ObjectPolarity','dark', 'Sensitivity',0.9); 

imshow(rgb) 
h = viscircles(centers,radii); 
[centers,radii] = imfindcircles(rgb,[20  25],'ObjectPolarity','dark', 'Sensitivity',0.92);
length(centers) 
delete(h)  % Delete previously drawn circles 

h = viscircles(centers,radii);
[centers,radii] = imfindcircles(rgb,[20  25],'ObjectPolarity','dark', 'Sensitivity',0.92,'Method','twostage');
delete(h) 

h = viscircles(centers,radii); 
[centers,radii] = imfindcircles(rgb,[20  25],'ObjectPolarity','dark', 'Sensitivity',0.95);
delete(h) 
viscircles(centers,radii); 
imshow(gray_image) 
[centersBright,radiiBright] = imfindcircles(rgb,[20 25], 'ObjectPolarity','bright','Sensitivity',0.92); 
imshow(rgb)

hBright = viscircles(centersBright, radiiBright,'Color','b'); 
[centersBright,radiiBright,metricBright] = imfindcircles(rgb,[20  25], 'ObjectPolarity','bright','Sensitivity',0.92,'EdgeThreshold',0.1);
delete(hBright) 
hBright = viscircles(centersBright, radiiBright,'Color','b'); 
h = viscircles(centers,radii);

%% Tahoe

figure(9) % Load images. 
buildingScene = imageDatastore('C:\Users\elick\Documents\MATLAB\image processing\Tahoe');
% Display images to be stitched
montage(buildingScene.Files) % Read the first image from the image set.
I = readimage(buildingScene, 1);

% Initialize features for I(1) 
grayImage = rgb2gray(I); 
points = detectSURFFeatures(grayImage); 
[features, points] = extractFeatures(grayImage, points);

% Initialize all the transforms to the identity matrix. Note that the 
% projective transform is used here because the building images are  fairly 
% close to the camera. Had the scene been captured from a further  distance, 
% an affine transform would suffice. 
numImages = numel(buildingScene.Files); 
tforms(numImages) = projective2d(eye(3));

% Initialize variable to hold image sizes. 
imageSize = zeros(numImages,2);
% Iterate over remaining image pairs 
for n = 2:numImages
    % Store points and features for I(n-1).
    pointsPrevious = points;     
    featuresPrevious = features;
    % Read I(n).
    I = readimage(buildingScene, n);
    % Convert image to grayscale.     
    grayImage = rgb2gray(I);
    % Save image size.     
    imageSize(n,:) = size(grayImage);
    % Detect and extract SURF features for I(n).
    points = detectSURFFeatures(grayImage);    
    [features, points] = extractFeatures(grayImage, points);
    % Find correspondences between I(n) and I(n-1).     
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique',  true);
    matchedPoints = points(indexPairs(:,1), :);     
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);
    % Estimate the transformation between I(n) and I(n-1).     
    tforms(n) = estimateGeometricTransform(matchedPoints,  matchedPointsPrev, 'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
    % Compute T(n) * T(n-1) * ... * T(1)     tforms(n).
    T = tforms(n).T * tforms(n-1).T; 
end

% Compute the output limits  for each transform 
for i = 1:3    
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1 imageSize(i,2)], [1 imageSize(i,1)]); 
end

avgXLim = mean(xlim, 2); 
[~, idx] = sort(avgXLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx);
Tinv = invert(tforms(centerImageIdx));

for i = 1:3     
    tforms(i).T = tforms(i).T * Tinv.T; 
end
for i = 1:3     
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1  imageSize(i,2)], [1 imageSize(i,1)]); 
end
maxImageSize = max(imageSize);

% Find the minimum and maximum output limits 
xMin = min([1; xlim(:)]); 
xMax = max([maxImageSize(2); xlim(:)]);
yMin = min([1; ylim(:)]); yMax = max([maxImageSize(1); ylim(:)]);
% Width and height of panorama. 
width  = round(xMax - xMin); 
height = round(yMax - yMin);
% Initialize the "empty" panorama. 
panorama = zeros([height width 3], 'like', I);
blender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');
% Create a 2-D spatial reference object defining the size of the  panorama.
xLimits = [xMin xMax]; 
yLimits = [yMin yMax]; 
panoramaView = imref2d([height width], xLimits, yLimits);
% Create the panorama. 
for i = 1:numImages    
    I = readimage(buildingScene, i);
    % Transform I into the panorama.     
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
    % Generate a binary mask.   
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView',  panoramaView);
    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask); 
end
figure
imshow(panorama)

%% Building Sample
figure(11) 
% Load images. 
buildingScene = imageDatastore('C:\Users\elick\Documents\MATLAB\image processing\Building');

% Display images to be stitched 
montage(buildingScene.Files)
% Read the first image from the image set.
I = readimage(buildingScene, 1);
% Initialize features for I(1) 
grayImage = rgb2gray(I); 
points = detectSURFFeatures(grayImage); 
[features, points] = extractFeatures(grayImage, points);
% Initialize all the transforms to the identity matrix. Note that the 
% projective transform is used here because the building images are  fairly 
% close to the camera. Had the scene been captured from a further  distance, 
% an affine transform would suffice. 

numImages = numel(buildingScene.Files); 
tforms(numImages) = projective2d(eye(3));
% Initialize variable to hold image sizes. 
imageSize = zeros(numImages,2);
% Iterate over remaining image pairs 
for n = 2:numImages
    % Store points and features for I(n-1).
    pointsPrevious = points;     
    featuresPrevious = features;
    % Read I(n).
    I = readimage(buildingScene, n);
    % Convert image to grayscale.     
    grayImage = rgb2gray(I);
    % Save image size.     
    imageSize(n,:) = size(grayImage);
    % Detect and extract SURF features for I(n).
    points = detectSURFFeatures(grayImage);     
    [features, points] = extractFeatures(grayImage, points);
    % Find correspondences between I(n) and I(n-1).     
    indexPairs = matchFeatures(features, featuresPrevious, 'Unique',  true);
    matchedPoints = points(indexPairs(:,1), :);     
    matchedPointsPrev = pointsPrevious(indexPairs(:,2), :);     
    % Estimate the transformation between I(n) and I(n-1).     
    tforms(n) = estimateGeometricTransform(matchedPoints,  matchedPointsPrev, 'projective', 'Confidence', 99.9, 'MaxNumTrials', 2000);
    % Compute T(n) * T(n-1) * ... * T(1)     
    tforms(n).T = tforms(n).T * tforms(n-1).T; 
end

% Compute the output limits  for each transform 
for i = 1:numel(tforms)     
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1  imageSize(i,2)], [1 imageSize(i,1)]); 
end

avgXLim = mean(xlim, 2); 
[~, idx] = sort(avgXLim);
centerIdx = floor((numel(tforms)+1)/2);
centerImageIdx = idx(centerIdx); 
Tinv = invert(tforms(centerImageIdx));

for i = 1:numel(tforms)     
    tforms(i).T = tforms(i).T * Tinv.T; 
end
for i = 1:numel(tforms)     
    [xlim(i,:), ylim(i,:)] = outputLimits(tforms(i), [1  imageSize(i,2)], [1 imageSize(i,1)]); 
end
maxImageSize = max(imageSize);

% Find the minimum and maximum output limits xMin = min([1; xlim(:)]); xMax = max([maxImageSize(2); xlim(:)]);
yMin = min([1; ylim(:)]); 
yMax = max([maxImageSize(1); ylim(:)]);
% Width and height of panorama. 
width  = round(xMax - xMin); 
height = round(yMax - yMin);
% Initialize the "empty" panorama. 
panorama = zeros([height width 3], 'like', I);
blender = vision.AlphaBlender('Operation', 'Binary mask', 'MaskSource', 'Input port');
% Create a 2-D spatial reference object defining the size of the  panorama.
xLimits = [xMin xMax]; 
yLimits = [yMin yMax]; 
panoramaView = imref2d([height width], xLimits, yLimits);

% Create the panorama. 
for i = 1:numImages     
    I = readimage(buildingScene, i);
    % Transform I into the panorama.     
    warpedImage = imwarp(I, tforms(i), 'OutputView', panoramaView);
    % Generate a binary mask.     
    mask = imwarp(true(size(I,1),size(I,2)), tforms(i), 'OutputView',  panoramaView);
    % Overlay the warpedImage onto the panorama.
    panorama = step(blender, panorama, warpedImage, mask); 
end
figure 
imshow(panorama)





























