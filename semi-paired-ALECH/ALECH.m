%function [B1, B2, B3] = ALECH(X, Y, L, param, XTest, YTest)
%function [B1X,B1Y, B2, B3] = ALECH(X, Y, XU, YU, L, param, XTest, YTest)
function [B1, B2, B3] = ALECH(X, Y, XU, YU, L,LU, param, XTest, YTest)
G= NormalizeFea(L,1);
[n, dX] = size(X);
dY = size(Y,2);
X = X'; Y=Y'; L=L';G=G';
XU=XU';YU=YU';
c = size(L,1);

nbits = param.nbits;
alpha = param.alpha;
beta = param.beta;
lamb = param.lamb;

theta=param.theta;
gamma=param.gamma;
delta=param.delta;

eta = param.eta;
%rand('seed', 2020);
sel_sample = Y(:,randsample(n, 2000),:);
[pcaW, ~] = eigs(cov(sel_sample'), nbits);
V = pcaW'*Y;
B = sign(V);
B(B==0) = -1;
P = eye(c);
W = (P*L*V')/(V*V'+lamb*eye(nbits));
On = ones(1,n);

DHamm = hammingDist(LU,L');
[~, orderH] = sort(DHamm, 2);
Or=orderH(:,1);
SX=Sset(850,Or,n);
SY=Sset(1200,Or,n);
% SX=Srand(n,850);
% SY=Srand(n,1200);
AX=pinv(SX'*(X'*X)*SX+delta*eye(850))*SX'*(X'*XU);
AY=pinv(SY'*(Y'*Y)*SY+delta*eye(1200))*SY'*(Y'*YU);


for iter = 1:param.iter  

   % B-step
    B = sign(2*nbits*alpha*V*G'*G - nbits*alpha*V*On'*On + beta*V);
 
    BUX=sign(V*SX*AX);
    BUY=sign(V*SY*AY);
   
    
    % P-step
    ZA = 2*(L*L') + eta*eye(c);
    ZB = W*V*L' + L*L';
    P = ZB/ZA;
    clear ZA ZB;

    % W-step
    W = (P*L*V')/(V*V'+ eta*eye(nbits));
    
    % V-step
      %Z = W'*P*L + beta*B + 2*alpha*nbits*B*G'*G - alpha*nbits*B*On'*On;
     Z = W'*P*L + beta*B + 2*alpha*nbits*B*(G'*G) - alpha*nbits*B*(On'*On)+theta*BUX*AX'*SX'+theta*BUY*AY'*SY';
     Z = Z';
     Temp = Z'*Z-1/n*(Z'*ones(n,1)*(ones(1,n)*Z));
     [~,Lmd,QQ] = svd(Temp); clear Temp
     idx = (diag(Lmd)>1e-4);
     Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
     Pt = (Z-1/n*ones(n,1)*(ones(1,n)*Z)) *  (Q / (sqrt(Lmd(idx,idx))));
     P_ = orth(randn(n,nbits-length(find(idx==1))));
     V = sqrt(n)*[Pt P_]*[Q Q_]';
     V = V';  

     %
   
end
%     BUX=sign(V*SX*AX);
%     BUY=sign(V*SY*AY);
%     BUX=BUX';
%     BUY=BUY';
%     BUX=BUX>0;
%     BUY=BUY>0;
    B1 = B';		
	B1 = B1>0;
%     B1X=[B1;BUX];
%     B1Y=[B1;BUY];
   
   % Gx = pinv(B*B'+lamb*eye(nbits))*(2*nbits*B*G'*G*X' - nbits*B*On'*On*X' +lamb*B*X')*pinv(X*X');
    Gxnew=lyap(pinv(gamma*eye(nbits))*(B*B'+lamb*eye(nbits)),(XU*XU')*pinv(X*X'),-(pinv(gamma*eye(nbits))*(2*nbits*B*(G'*G)*X' - nbits*B*(On'*On)*X' +lamb*B*X'+gamma*V*SX*AX*XU')*pinv(X*X')));
   % B2 = XTest*Gx'>0;
    B2 = XTest*Gxnew'>0;
    
    
    %Gy = pinv(B*B'+lamb*eye(nbits))*(2*nbits*B*G'*G*Y'- nbits*B*On'*On*Y'+lamb*B*Y')*pinv(Y*Y');
    Gynew=lyap(pinv(gamma*eye(nbits))*(B*B'+lamb*eye(nbits)),(YU*YU')*pinv(Y*Y'),-(pinv(gamma*eye(nbits))*(2*nbits*B*(G'*G)*Y' - nbits*B*(On'*On)*Y' +lamb*B*Y'+gamma*V*SY*AY*YU')*pinv(Y*Y')));
    %B3 = YTest*Gy'>0;
    B3 = YTest*Gynew'>0;
   
  
    
end

