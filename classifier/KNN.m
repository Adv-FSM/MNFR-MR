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
 Mdl_KNN = fitcknn(...
    df, ...
    f, ...
    'Distance', 'Minkowski', ...
    'Exponent', 3, ...
    'NumNeighbors', 10, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true);
 
partitionedModel_KNN = crossval(Mdl_KNN,'KFold', 5);

validationAccuracy_KNN(i) = 1 - kfoldLoss(partitionedModel_KNN);

end

X_KNN=mean(validationAccuracy_KNN)


