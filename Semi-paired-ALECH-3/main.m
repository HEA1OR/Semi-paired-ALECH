load('IAPRTC-12.mat');
LUTrain= L_tr(10001:14000,:);
LTrain=L_tr(1:10000,:);
XUTrain= I_tr(10001:14000,:);
XTrain=I_tr(1:10000,:);
YUTrain= T_tr(10001:14000,:);
YTrain=T_tr(1:10000,:);
VUTrain= V_tr(10001:14000,:);
VTrain=V_tr(1:10000,:);
LTest=L_te;
XTest=I_te;
YTest=T_te;
VTest=V_te;

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
[VTest,VTrain,VUTrain]=centralize(VTest,VTrain,VUTrain);
%% kernelization
[XKTrain,XKTest,XKUTrain] = Kernelize(XTrain, XTest, XUTrain,500);
[YKTrain,YKTest,YKUTrain]=Kernelize(YTrain,YTest,YUTrain, 2000);
[VKTrain,VKTest,VKUTrain]=Kernelize(VTrain,VTest,VUTrain, 3000);
[XKTest,XKTrain,XKUTrain]=centralize(XKTest,XKTrain,XKUTrain);
[YKTest,YKTrain,YKUTrain]=centralize(YKTest,YKTrain,YKUTrain);
[VKTest,VKTrain,VKUTrain]=centralize(VKTest,VKTrain,VKUTrain);
%% ALECH
for kk= 1:length(nbitset)
% [XKTrain,XKTest,XKUTrain] = Kernelize(XTrain, XTest, XUTrain,500); [YKTrain,YKTest,YKUTrain]=Kernelize(YTrain,YTest,YUTrain, 1000);
% % [XKTrain,XKTest,XKUTrain] = Kernelize(XTrain, XTest, XUTrain,484+1*nbitset(kk)); [YKTrain,YKTest,YKUTrain]=Kernelize(YTrain,YTest,YUTrain, 984+1*nbitset(kk));
% [XKTest,XKTrain,XKUTrain]=centralize(XKTest,XKTrain,XKUTrain);
% [YKTest,YKTrain,YKUTrain]=centralize(YKTest,YKTrain,YKUTrain);
param.nbits = nbitset(kk);


%[B1X,B1Y, B2, B3] = ALECH(XKTrain, YKTrain, XKUTrain, YKUTrain, LTrain, param, XKTest, YKTest);
[B1, B2, B3, B4] = ALECH(XKTrain, YKTrain, VKTrain,XKUTrain, YKUTrain, VKUTrain, LTrain,LUTrain, param, XKTest, YKTest, VKTest);


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