% Induction Machine Dynamics
%for steady state evaluation
function xdot=imdyn_ss(x,z)
k1=-0.186; k2=0.178; k3= 0.225; k4=-0.234; k5=-0.081; k6=4.643;
z(1)=1;z(2)=1;z(3)=0; %substituted the input values
xdot(1)=k1*x(1)+z(1)*x(2)+k2*x(3)+z(2);
xdot(2)=-z(1)*x(1)+k1*x(2)+k2*x(4);
xdot(3)=k3*x(1)+k4*x(3)+x(4)*(z(1)-x(5));
xdot(4)=k3*x(2)-x(3)*(z(1)-x(5))+k4*x(4);
xdot(5)=k5*(x(1)*x(4)-x(2)*x(3))+k6*z(3);
%xdot=xdot';
end







% %run the below line in cmd
% fun = @imdyn_ss;
% x0 = [0 0 0 1 1];
% x = fsolve(fun,x0)
% %check the correctness of sol
% feval(fun,[ans]
%xeq= [0.0148   -0.9998    0.0143   -0.9613    1.0000]