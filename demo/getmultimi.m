function c = getmultimi(da, dt) 
for i=1:size(da,2) 
   c(i) = mutualinfo(da(:,i), dt); 
end