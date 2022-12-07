function [output] = Srand(n,m)
   %rand_pos gets a random number in range [1, n].
A=zeros(n,m);
B=zeros(n);
for i=1:m
    
    rand_pos = randi([1, n]);
  
%     while(B(rand_pos)==1)
%          rand_pos = randi([1, n]);
%     end
     A(rand_pos, i)=1;
        B(rand_pos)=1;
        
end
output=A;
end

