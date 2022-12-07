load('flickr.mat');
LUTrain= LTrain(8001:12000,:);
LTrain=LTrain(1:10000,:);
XUTrain= XTrain(8001:12000,:);
XTrain=XTrain(1:10000,:);
YUTrain= YTrain(8001:12000,:);
YTrain=YTrain(1:10000,:);

%% parameter set
param.alpha = 1e-4;
param.beta = 1e-3; 
param.lamb  = 1e4;
param.eta  = 1e-4;
param.iter  = 13;
%待调参数
param.gamma=1e-3;
param.delta=1e-3;
param.theta=1e-2;
% lt=Srand(8015,1000);


% LUTrain=lt'*LUTrain;
% LTrain=[LTrain;LUTrain];
nbitset     = [8,16,32,64,128];
eva_info    = cell(1,length(nbitset));
%% centralization
[XTest,XTrain,XUTrain]=centralize(XTest,XTrain,XUTrain);
[YTest,YTrain,YUTrain]=centralize(YTest,YTrain,YUTrain);

%% kernelization
[XKTrain,XKTest,XKUTrain] = Kernelize(XTrain, XTest, XUTrain,500); [YKTrain,YKTest,YKUTrain]=Kernelize(YTrain,YTest,YUTrain, 1000);
[XKTest,XKTrain,XKUTrain]=centralize(XKTest,XKTrain,XKUTrain);
[YKTest,YKTrain,YKUTrain]=centralize(YKTest,YKTrain,YKUTrain);
%% ALECH
for kk= 1:length(nbitset)
% [XKTrain,XKTest,XKUTrain] = Kernelize(XTrain, XTest, XUTrain,500); [YKTrain,YKTest,YKUTrain]=Kernelize(YTrain,YTest,YUTrain, 1000);
% % [XKTrain,XKTest,XKUTrain] = Kernelize(XTrain, XTest, XUTrain,484+1*nbitset(kk)); [YKTrain,YKTest,YKUTrain]=Kernelize(YTrain,YTest,YUTrain, 984+1*nbitset(kk));
% [XKTest,XKTrain,XKUTrain]=centralize(XKTest,XKTrain,XKUTrain);
% [YKTest,YKTrain,YKUTrain]=centralize(YKTest,YKTrain,YKUTrain);
param.nbits = nbitset(kk);


%[B1X,B1Y, B2, B3] = ALECH(XKTrain, YKTrain, XKUTrain, YKUTrain, LTrain, param, XKTest, YKTest);
[B1, B2, B3] = ALECH(XKTrain, YKTrain, XKUTrain, YKUTrain, LTrain,LUTrain, param, XKTest, YKTest);


%DHamm = hammingDist(B2, B1X);
DHamm = hammingDist(B2, B1);
[~, orderH] = sort(DHamm, 2);
%LS=[LTrain;LUTrain];
%eva_info_.Image_to_Text_MAP = mAP(orderH', LS, LTest);
 eva_info_.Image_to_Text_MAP = mAP(orderH', LTrain, LTest);
 
%DHamm = hammingDist(B3, B1Y);
DHamm = hammingDist(B3, B1);
[~, orderH] = sort(DHamm, 2);
%eva_info_.Text_to_Image_MAP = mAP(orderH', LS, LTest);
 eva_info_.Text_to_Image_MAP = mAP(orderH', LTrain, LTest);

eva_info{1,kk} = eva_info_;
Image_to_Text_MAP = eva_info_.Image_to_Text_MAP;
Text_to_Image_MAP = eva_info_.Text_to_Image_MAP;  

fprintf('ALECH %d bits -- Image_to_Text_MAP: %.4f ; Text_to_Image_MAP: %.4f ; \n',nbitset(kk),Image_to_Text_MAP,Text_to_Image_MAP);
end