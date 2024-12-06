clc;
clear all;

% Replace your file
mat = load('...\dataset\HDSS\COIL20');
d = mat.('X');
f = mat.('Y');

% Replace your features
Q=d(:,[446	556	362	217	390	483	710	447	489	348	588	358	649	477	249	488	678	511	451	143	329	524	547	250	414	422	179	553	478	784]);

nd=size(Q,2);

for i=1:nd
 df=Q(:,[1:i]);
 Mdl_NN = fitcnet(...
    df, ...
    f, ...
    'LayerSizes', [10 10], ...
    'Activations', 'relu', ...
    'Lambda', 0, ...
    'IterationLimit', 1000, ...
    'Standardize', true);
 
partitionedModel = crossval(Mdl_NN,'KFold', 5);

validationAccuracy(i) = 1 - kfoldLoss(partitionedModel);

end

X_NN=mean(validationAccuracy)


