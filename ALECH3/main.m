%load('flickr.mat');
load('IAPRTC-12.mat');
LTrain=L_tr;
XTrain=I_tr;
YTrain=T_tr;
VTrain=V_tr;
LTest=L_te;
XTest=I_te;
YTest=T_te;
VTest=V_te;
%% parameter set
param.alpha = 1e-4;
param.beta = 1e-3; 
param.lamb  = 1e4;
param.eta  = 1e-4;
param.iter  = 10;
nbitset     = [8,16,32,64,128];
eva_info    = cell(1,length(nbitset));
%% centralization
XTest = bsxfun(@minus, XTest, mean(XTrain, 1)); XTrain = bsxfun(@minus, XTrain, mean(XTrain, 1));
YTest = bsxfun(@minus, YTest, mean(YTrain, 1)); YTrain = bsxfun(@minus, YTrain, mean(YTrain, 1));
VTest = bsxfun(@minus, VTest, mean(VTrain, 1)); VTrain = bsxfun(@minus, VTrain, mean(VTrain, 1));
%% kernelization
[XKTrain,XKTest] = Kernelize(XTrain, XTest, 500); [YKTrain,YKTest]=Kernelize(YTrain,YTest, 2000);
[VKTrain,VKTest]=Kernelize(VTrain,VTest, 3000);
XKTest = bsxfun(@minus, XKTest, mean(XKTrain, 1)); XKTrain = bsxfun(@minus, XKTrain, mean(XKTrain, 1));
YKTest = bsxfun(@minus, YKTest, mean(YKTrain, 1)); YKTrain = bsxfun(@minus, YKTrain, mean(YKTrain, 1));
VKTest = bsxfun(@minus, VKTest, mean(VKTrain, 1)); VKTrain = bsxfun(@minus, VKTrain, mean(VKTrain, 1));
%% ALECH
for kk= 1:length(nbitset)
    
param.nbits = nbitset(kk);

[B1, B2, B3, B4] = ALECH(XKTrain, YKTrain, VKTrain, LTrain, param, XKTest, YKTest, VKTest);

DHamm = hammingDist(B2, B1);
[~, orderH] = sort(DHamm, 2);
eva_info_.Image_MAP = mAP(orderH', LTrain, LTest);
 
DHamm = hammingDist(B3, B1);
[~, orderH] = sort(DHamm, 2);
eva_info_.Text_MAP = mAP(orderH', LTrain, LTest);

DHamm = hammingDist(B4, B1);
[~, orderH] = sort(DHamm, 2);
eva_info_.Video_MAP = mAP(orderH', LTrain, LTest);

eva_info{1,kk} = eva_info_;
Image_MAP = eva_info_.Image_MAP;
Text_MAP = eva_info_.Text_MAP;  
Video_MAP = eva_info_.Video_MAP;  

fprintf('ALECH %d bits -- Image_MAP: %.4f ; Text_MAP: %.4f ; Video_MAP: %.4f ; \n',nbitset(kk),Image_MAP,Text_MAP,Video_MAP);
end