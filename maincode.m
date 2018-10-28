% main code for induction machine project

randn('state',1)
k7= -4.448; k8=1;
N=500; Ts=0.1;
x_initial=[0.2 -0.6 -0.4 0.1 0.3];
x(:,1)=x_initial; % initial condition
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
subplot(321),plot(T,x(1,:),T,xkk(1,:)), ylabel('x_1 (flux)'), title('KF'),legend('true', 'estimated')
subplot(322),plot(T,x(2,:),T,xkk(2,:)), ylabel('x_2 (flux)'), title('KF'),legend('true', 'estimated')
subplot(323),plot(T,x(3,:),T,xkk(3,:)), ylabel('x_3 (flux)'), title('KF'),legend('true', 'estimated')
subplot(324),plot(T,x(4,:),T,xkk(4,:)), ylabel('x_4 (flux)'), title('KF'),legend('true', 'estimated')
subplot(325),plot(T,x(5,:),T,xkk(5,:)), ylabel('x_5 (angular velocity)'), title('KF'),legend('true', 'estimated')

%% Implementation of EKF
% measurement generation for EKF
for i=1:N
    ey(:,i)=C*x(:,i) + vk(i,:)' ; %measurement in non-delta y form
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
subplot(321),plot(T,x(1,:),T,exkk(1,:)), ylabel('x_1 (flux)'), title('EKF'),legend('true', 'estimated')
subplot(322),plot(T,x(2,:),T,exkk(2,:)), ylabel('x_2 (flux)'), title('EKF'),legend('true', 'estimated')
subplot(323),plot(T,x(3,:),T,exkk(3,:)), ylabel('x_3 (flux)'), title('EKF'),legend('true', 'estimated')
subplot(324),plot(T,x(4,:),T,exkk(4,:)), ylabel('x_4 (flux)'), title('EKF'),legend('true', 'estimated')
subplot(325),plot(T,x(5,:),T,exkk(5,:)), ylabel('x_5 (angular velocity)'), title('EKF'),legend('true', 'estimated')

%% Implementation of UKF
M= 5 + length(mw) + length(mv);
Ns=2*M+1;
uPkk(:,:,1)=blkdiag(eye(5),Q,R); %combined covariance matrix
chikk(:,1)=[x_initial';mw';mv'];%combinded state vector
chikk_i(:,1)=chikk(:,1);
kappa=1; %tuning parameter
rho=sqrt(M+kappa);
omega_i(:,1)=kappa/(M+kappa);
xhat_kk(:,1)=x_initial';
for k =1:N
    for i=1:M  
        % Sigma point generation
        zeta_i=zeros(M,1);
        zeta_i(i)=1;
        chikk_i(:,i+1)=chikk(:,k)+ rho*chol(uPkk(:,:,k))*zeta_i;
        chikk_i(:,i+M+1)=chikk(:,k)- rho*sqrtm(uPkk(:,:,k))*zeta_i;
        % weights generation
        omega_i(:,i+1)=1/(2*(M+kappa));
        omega_i(:,i+M+1)=1/(2*(M+kappa));
    end
    % Propagation of samples through system dynamics
    x_sample_mean=zeros(5,1);
    Y_sample_mean=zeros(2,1);
    
    for j=1:Ns
        % Runge Kutta Integration 2nd order
        x_tilda_k=xhat_kk(:,k) + (Ts/2.0)*imdyn(1,chikk_i(1:5,j),u);
        xhat_kk1_i(:,j)=xhat_kk(:,k) + Ts*imdyn(1,x_tilda_k,u) + chikk_i(6:10,j);
%         xhat_kk1_i(:,j)=xhat_kk(:,k) + Ts*imdyn(1,chikk_i(1:5,j),u) + chikk_i(6:10,j);
%         [t,xt]=ode45(@(t,x)imdyn(t,x,u),[0 Ts],chikk_i(1:5,j));
%         xhat_kk1_i(:,j) =xhat_kk(:,k)+ xt(end,:)' + chikk_i(6:10,j);
        x_sample_mean=x_sample_mean + omega_i(:,j)*xhat_kk1_i(:,j);
        Y_i(:,j)=C*xhat_kk1_i(:,j) + chikk_i(11:12,j);
        Y_sample_mean=Y_sample_mean + omega_i(:,j)*Y_i(:,j);
    end
    xhat_kk1(:,k)=x_sample_mean;
    Yhat_kk1(:,k)=Y_sample_mean;
    %covariance variable declaration
    PEEkk1=zeros(5);
    PEekk=zeros(5,2);
    Peekk=zeros(2);
    for p=1:Ns
        %error calculation
        e_i(:,p)= Y_i(:,p) -  Yhat_kk1(:,k);
        E_i(:,p)= xhat_kk1_i(:,p) - xhat_kk1(:,k);
        % covariances calculation
        PEEkk1 = PEEkk1 + omega_i(:,p)*E_i(:,p)*E_i(:,p)'; 
        PEekk= PEekk + omega_i(:,p)*E_i(:,p)*e_i(:,p)';
        Peekk= Peekk + omega_i(:,p)*e_i(:,p)*e_i(:,p)';
    end
    %Kalman Gain Update     
    Lk=PEekk*inv(Peekk);
    e_u(:,k)= ey(:,k) - Yhat_kk1(:,k) ; % innovation % measurement variable ey is taken from above (near EKF)
    xhat_kk(:,k+1)=xhat_kk1(:,k) + Lk*e_u(:,k) ; % state update
    Pkk(:,:,k+1) = PEEkk1 - Lk*Peekk*Lk';  %covariance update
    uPkk(:,:,k+1)=blkdiag(Pkk(:,:,k+1),Q,R); % the big combined covariance matrix update
    chikk(:,k+1)= [xhat_kk(:,k+1);mw';mv'];
    chikk_i(:,1)=chikk(:,k+1); % first element of chi updated
end
%% plots for UKF
figure(3)
subplot(321),plot(T,x(1,:),T,xhat_kk(1,:)), ylabel('x_1 (flux)'), title('UKF'),legend('true', 'estimated')
subplot(322),plot(T,x(2,:),T,xhat_kk(2,:)), ylabel('x_2 (flux)'), title('UKF'),legend('true', 'estimated')
subplot(323),plot(T,x(3,:),T,xhat_kk(3,:)), ylabel('x_3 (flux)'), title('UKF'),legend('true', 'estimated')
subplot(324),plot(T,x(4,:),T,xhat_kk(4,:)), ylabel('x_4 (flux)'), title('UKF'),legend('true', 'estimated')
subplot(325),plot(T,x(5,:),T,xhat_kk(5,:)), ylabel('x_5 (angular velocity)'), title('UKF'),legend('true', 'estimated')
%% plotting of true, KF, EKF, and UKF
figure(4)
subplot(321),plot(T,x(1,:),T,xkk(1,:),T,exkk(1,:),T,xhat_kk(1,:)), ylabel('x_1 (flux)'), title('True vs estimated'),legend('true', 'KF','EKF','UKF')
subplot(322),plot(T,x(2,:),T,xkk(2,:),T,exkk(2,:),T,xhat_kk(2,:)), ylabel('x_2 (flux)'), title('True vs estimated'),legend('true', 'KF','EKF','UKF')
subplot(323),plot(T,x(3,:),T,xkk(3,:),T,exkk(3,:),T,xhat_kk(3,:)), ylabel('x_3 (flux)'), title('True vs estimated'),legend('true', 'KF','EKF','UKF')
subplot(324),plot(T,x(4,:),T,xkk(4,:),T,exkk(4,:),T,xhat_kk(4,:)), ylabel('x_4 (flux)'), title('True vs estimated'),legend('true', 'KF','EKF','UKF')
subplot(325),plot(T,x(5,:),T,xkk(5,:),T,exkk(5,:),T,xhat_kk(5,:)), ylabel('x_5 (angular velocity)'), title('True vs estimated'),legend('true', 'KF','EKF','UKF')
%% plotting of EKF and UKF on same plot
figure(5)
subplot(321),plot(T,x(1,:),T,exkk(1,:),T,xhat_kk(1,:)), ylabel('x_1 (flux)'), title('True vs estimated'),legend('true', 'EKF','UKF')
subplot(322),plot(T,x(2,:),T,exkk(2,:),T,xhat_kk(2,:)), ylabel('x_2 (flux)'), title('True vs estimated'),legend('true', 'EKF','UKF')
subplot(323),plot(T,x(3,:),T,exkk(3,:),T,xhat_kk(3,:)), ylabel('x_3 (flux)'), title('True vs estimated'),legend('true', 'EKF','UKF')
subplot(324),plot(T,x(4,:),T,exkk(4,:),T,xhat_kk(4,:)), ylabel('x_4 (flux)'), title('True vs estimated'),legend('true','EKF','UKF')
subplot(325),plot(T,x(5,:),T,exkk(5,:),T,xhat_kk(5,:)), ylabel('x_5 (angular velocity)'), title('True vs estimated'),legend('true','EKF','UKF')
%% innovation plot for KF and UKF
figure(6)
subplot(211),plot(T(2:end),e(1,:),T(2:end),e_u(1,:)), ylabel('x_1 (flux)'), title('innovation'),legend( 'EKF','UKF')
subplot(212),plot(T(2:end),e(2,:),T(2:end),e_u(2,:)), ylabel('x_2 (flux)'), title('innovation'),legend( 'EKF','UKF')

