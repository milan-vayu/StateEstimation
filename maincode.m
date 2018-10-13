% main code for induction machine project

randn('state',1)
k7= -4.448; k8=1;
N=500; Ts=0.1;
x(:,1)=[0.2 -0.6 -0.4 0.1 0.3]; % initial condition
u=[1 1 0];  % constant input 
Q=(1e-4)*eye(5);mw=[0 0 0 0 0];
R=(1e-2)*eye(2); mv=[0 0];
T(:,1)=0;
%% True state generation
for i=1:N
    [t,xt]=ode45(@(t,x)imdyn(t,x,u),[0 Ts],x(:,i));
    x(:,i+1)=xt(end,:)+ mvnrnd(mw,Q); %process noise added after integration
    T(:,i+1)=Ts*(i);
end

%% Finding Stedy state operating point  
fun = @imdyn_ss; % function with substituted Z(input) values
x0 = [0 0 0 1 1]; %  point around which fsolve will start search for steady state point
xss = fsolve(fun,x0); % xss= [0.0148   -0.9998    0.0143   -0.9613    1.0000]
%To check the correctness of solution run 'feval'
%feval(fun,xss)  % To check the correctness of steady state point
%jacobian calculation---------
A=jacob(@(x)imdyn(1,x,u),xss);
B=jacob(@(z) imdyn(1,xss,z),u);
%% Linear Discrete time model
phi=expm(A*Ts);
gama=(phi-eye(5))*inv(A)*B;
%measurement matrix, c
C=[k7 0 k8 0 0;0 k7 0 k8 0];
%x(k+1)=phi*x(k)+gama*u+w(k)
%% measurement generation
vk=mvnrnd(mv,R,N);
for i=1:N
    y(:,i)=C*(x(:,i)-xss') + vk(i,:)' ; %measurement in delta y form
end

%% implementation of KF
xkk(:,1)=x(:,1);  %given initial state
Pkk(:,:,1)=eye(5); %given initial covariance
for i=1:N
    xkk1(:,i)=phi*xkk(:,i) + gama*u';
    Pkk1(:,:,i)=phi*Pkk(:,:,i)*phi' + Q ;% assumed gamad =I
    Lk=Pkk1(:,:,i)*C'*inv(C*Pkk1(:,:,i)*C' + R);
    e(:,i)=y(:,i)-C*xkk1(:,i);
    xkk(:,i+1)=xkk1(:,i)+Lk*e(:,i);
    Pkk(:,:,i+1)=(eye(5)-Lk*C)*Pkk1(:,:,1);
end
%% plots for KF
figure(1)
subplot(321),plot(T,x(1,:),T,xkk(1,:)), ylabel('x_1 (flux)'), title('State Variables'),legend('true', 'estimated')
subplot(322),plot(T,x(2,:),T,xkk(2,:)), ylabel('x_2 (flux)'), title('State Variables'),legend('true', 'estimated')
subplot(323),plot(T,x(3,:),T,xkk(3,:)), ylabel('x_3 (flux)'), title('State Variables'),legend('true', 'estimated')
subplot(324),plot(T,x(4,:),T,xkk(4,:)), ylabel('x_4 (flux)'), title('State Variables'),legend('true', 'estimated')
subplot(325),plot(T,x(5,:),T,xkk(5,:)), ylabel('x_5 (angular velocity)'), title('State Variables'),legend('true', 'estimated')

%% Implementation of EKF
% measurement generation for EKF
for i=1:N
    ey(:,i)=C*x(:,i) + vk(i,:)' ; %measurement in delta y form
end
exkk(:,1)=0.9*x(:,1); %0.9*initial state 
ePkk(:,:,1)=eye(5);

for i=1:N
    % Succesive Linearization
    A=jacob(@(x)imdyn(1,x,u),exkk(:,i));
    Bd=eye(5) ; %since noise was individually added 
    phi=expm(A*Ts);
    gamad=eye(5); % (phi-eye(5))*inv(A)*Bd;
    % Prediction step
    exkk1(:,i)=exkk(:,i)+Ts*imdyn(1,exkk(:,i),u);
    ePkk1(:,:,i)=phi*ePkk(:,:,i)*phi' +gamad*Q*gamad';
    % Kalman gain computation
    Lk=ePkk1(:,:,i)*C'*inv(C*ePkk1(:,:,i)*C'+R);
    % Update step
    e(:,i)=ey(:,i)-C*exkk1(:,i);
    exkk(:,i+1)=exkk1(:,i) + Lk*e(:,i);
    ePkk(:,:,i+1)=(eye(5)-Lk*C)*ePkk1(:,:,i);
end
%% plots for EKF
figure(2)
subplot(321),plot(T,x(1,:),T,exkk(1,:)), ylabel('x_1 (flux)'), title('State Variables'),legend('true', 'estimated')
subplot(322),plot(T,x(2,:),T,exkk(2,:)), ylabel('x_2 (flux)'), title('State Variables'),legend('true', 'estimated')
subplot(323),plot(T,x(3,:),T,exkk(3,:)), ylabel('x_3 (flux)'), title('State Variables'),legend('true', 'estimated')
subplot(324),plot(T,x(4,:),T,exkk(4,:)), ylabel('x_4 (flux)'), title('State Variables'),legend('true', 'estimated')
subplot(325),plot(T,x(5,:),T,exkk(5,:)), ylabel('x_5 (angular velocity)'), title('State Variables'),legend('true', 'estimated')
    
    

    
    


