ref_ID=zeros(96,1);
loc_ID=zeros(96,1);
A=randsample(linspace(0,11,12),8);
B=datasample([0,1],8);
for i=1:8
    for j=1:4
        for k=1:3
            ref_ID((i-1)*12+(j-1)*3+k)=(i-1)*12+(j-1)*3+k-1;
            loc_ID((i-1)*12+(j-1)*3+k)=i-1;
            tech_type((i-1)*12+(j-1)*3+k)=j;
            capacity((i-1)*12+(j-1)*3+k)=k*100;
            community_ID((i-1)*12+(j-1)*3+k)=A(i);
            co_oper((i-1)*12+(j-1)*3+k)=B(i);
            if capacity((i-1)*12+(j-1)*3+k)>100
                IRR_min((i-1)*12+(j-1)*3+k)=0.15;
            else
                IRR_min((i-1)*12+(j-1)*3+k)=0.1;
            end
        end
    end
end

loc_farm_dist=rand(8,300)*120+0.5;
ref_farm_dist=rand(96,300);
for i=1:96
    ID_loc_temp=loc_ID(i);
    ref_farm_dist(i,:)=loc_farm_dist(ID_loc_temp+1,:);
end

loc_loc_dist=exprnd(50,8,8);
for i=1:8
    loc_loc_dist(i,i)=0;
    for j=i:8
        loc_loc_dist(j,i)=loc_loc_dist(i,j);
    end
end

ref_ref_dist=zeros(96,96);
for i=1:96
    for j=1:96
        ref_ref_dist(i,j)=loc_loc_dist(loc_ID(i)+1,loc_ID(j)+1);
    end
end

