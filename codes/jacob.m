% jacobian calculation function
function gradf=jacob(f,z0)
h=1e-4;
for i=1:length(z0)
    zp=z0;
    zp(i)=z0(i)+h;
    fp=feval(f,zp);
    zn=z0;
    zn(i)=z0(i)-h;
    fn=feval(f,zn);
    gradf(:,i)=(fp-fn)/(2*h);
end
end