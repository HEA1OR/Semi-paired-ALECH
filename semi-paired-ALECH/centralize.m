function [XTest,XTrain,XUTrain]=centralize(A,B,C)
XS=[B;C];
XTest = bsxfun(@minus, A, mean(XS, 1)); XTrain = bsxfun(@minus, B, mean(XS, 1));
XUTrain = bsxfun(@minus,C, mean(XS, 1));

end

