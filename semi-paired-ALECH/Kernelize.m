function [KTrain, KTest,KUtrain] = Kernelize(Train,Test,Utrain,n_anchor)
    [x,~]=size(Train);
    [y,~]=size(Utrain);
    XU=[Train;Utrain];
    [n,~]=size(XU);      %行数
    [nT,~]=size(Test);
    anchor=XU(randsample(n,n_anchor),:);
   
    rand('seed', 2020);
    KU = sqdist(XU',anchor');
    sigma = mean(mean(KU,2));
    KU = exp(-KU/(2*sigma));  
    mvec = mean(KU);
    KU = KU-repmat(mvec,n,1);
    KTrain=KU(1:x,:);
    KUtrain=KU(x+1:x+y,:);
    
    KTest = sqdist(Test',anchor');
    KTest = exp(-KTest/(2*sigma));
    KTest = KTest-repmat(mvec,nT,1);
end