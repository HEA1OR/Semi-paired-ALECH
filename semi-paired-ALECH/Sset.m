function [M] = Sset(n,order,num)
    m=size(order,1);
    M=zeros(num,n);
    for i=1:n
        M(order(i),i)=1;
    end
end

