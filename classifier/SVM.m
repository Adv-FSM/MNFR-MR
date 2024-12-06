clc;
clear all;

% Replace your file
mat = load('...\dataset\HDSS\COIL20');
d = mat.('X');
f = mat.('Y');

% Replace your features
Q=d(:,[446	556	362	217	390	483	710	447	489	348	588	358	649	477	249	488	678	511	451	143	329	524	547	250	414	422	179	553	478	784]);

nd=size(Q,2);

template = templateSVM(...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true);
for i=1:nd
df=Q(:,[1:i]);
Mdl_SVM = fitcecoc(...
    df, ...
    f, ...
    'Learners', template, ...
    'Coding', 'onevsone');

partitionedModel_SVM = crossval(Mdl_SVM,'KFold', 5);

validationAccuracy_SVM(i) = 1 - kfoldLoss(partitionedModel_SVM);
end

X_SVM=mean(validationAccuracy_SVM)



