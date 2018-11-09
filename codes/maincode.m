% main code for induction machine project
clear; close all
randn('state',8)
k7= -4.448; k8=1;
N=500; Ts=0.1;
x_initial=[0.2 -0.6 -0.4 0.1 0.3];
%x_paper = [0.5 0.1 0.3 -0.2 4]; % initial estimate of x used in the paper by Kandepu
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

%% Finding Steady state operating point  
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
    spec_rad_KF_Pkk(i)=max(abs(eig(Pkk(:,:,i)))); % spectral radii of updated covariance
    spec_rad_KF_Pkk1(i)=max(abs(eig(Pkk1(:,:,i)))); % spectral radii of predicted covariance
    betak_KF(:,i)=(x(:,i)-xkk(:,i))'*inv(Pkk(:,:,i))*(x(:,i)-xkk(:,i)); %NESS calculation
end
%% plots for KF
figure(1)
subplot(321),plot(T,x(1,:),T,xkk(1,:)), ylabel('x_1 (flux)'), title('KF'),legend('true', 'estimated')
subplot(322),plot(T,x(2,:),T,xkk(2,:)), ylabel('x_2 (flux)'), title('KF'),legend('true', 'estimated')
subplot(323),plot(T,x(3,:),T,xkk(3,:)), ylabel('x_3 (flux)'), title('KF'),legend('true', 'estimated')
subplot(324),plot(T,x(4,:),T,xkk(4,:)), ylabel('x_4 (flux)'), title('KF'),legend('true', 'estimated')
subplot(325),plot(T,x(5,:),T,xkk(5,:)), ylabel('x_5 (angular velocity)'), title('KF'),legend('true', 'estimated')
close;
%% Implementation of EKF
% measurement generation for EKF
for i=1:N
    ey(:,i)=C*x(:,i) + vk(i,:)' ; %measurement in non-delta y form
end
exkk(:,1)=0.9*x_initial; %0.9*initial state 
ePkk(:,:,1)=eye(5);

for i=1:N
    % Succesive Linearization
    A=jacob(@(x)imdyn(1,x,u),exkk(:,i));
    Bd=eye(5) ; %since noise was individually added 
    phi=expm(A*Ts);
    gamad=eye(5); % (phi-eye(5))*inv(A)*Bd;
    % Prediction step
    exkk1(:,i)=exkk(:,i)+Ts*imdyn(1,exkk(:,i),u);
    ePkk1(:,:,i)=phi*ePkk(:,:,i)*phi' + Q ;
    % Kalman gain computation
    Lk=ePkk1(:,:,i)*C'*inv(C*ePkk1(:,:,i)*C'+R);
    % Update step
    E(:,i)=ey(:,i)-C*exkk1(:,i); % innovation
    exkk(:,i+1)=exkk1(:,i) + Lk*E(:,i);
    ePkk(:,:,i+1)=(eye(5)-Lk*C)*ePkk1(:,:,i);
    spec_rad_EKF_Pkk(i)=max(abs(eig(ePkk(:,:,i)))); % spectral radii of updated covariance
    spec_rad_EKF_Pkk1(i)=max(abs(eig(ePkk1(:,:,i)))); % spectral radii of predicted covariance
    betak_EKF(:,i)=(x(:,i)-exkk(:,i))'*inv(ePkk(:,:,i))*(x(:,i)-exkk(:,i)); % NESS calculation
end
%% plots for EKF
figure(2)
subplot(321),plot(T,x(1,:),T,exkk(1,:)), ylabel('x_1 (flux)'), title('EKF'),legend('true', 'estimated')
subplot(322),plot(T,x(2,:),T,exkk(2,:)), ylabel('x_2 (flux)'), title('EKF'),legend('true', 'estimated')
subplot(323),plot(T,x(3,:),T,exkk(3,:)), ylabel('x_3 (flux)'), title('EKF'),legend('true', 'estimated')
subplot(324),plot(T,x(4,:),T,exkk(4,:)), ylabel('x_4 (flux)'), title('EKF'),legend('true', 'estimated')
subplot(325),plot(T,x(5,:),T,exkk(5,:)), ylabel('x_5 (angular velocity)'), title('EKF'),legend('true', 'estimated')
close
%% Implementation of UKF
M= 5 + length(mw) + length(mv);
Ns=2*M+1;
uPakk(:,:,1)=blkdiag(eye(5),Q,R); %combined covariance matrix
chikk(:,1)=[0.9*x_initial';mw';mv'];%combinded state vector
chikk_i(:,1)=chikk(:,1);
kappa=3-M; %tuning parameter
rho=sqrt(M+kappa);
omega_i(:,1)=kappa/(M+kappa);
xhat_kk(:,1)=0.9*x_initial;
uPkk(:,:,1)=eye(5);
for k =1:N
    for i=1:M  
        % Sigma point generation
        zeta_i=zeros(M,1);
        zeta_i(i)=1;
        chikk_i(:,i+1)=chikk(:,k)+ rho*chol(uPakk(:,:,k))*zeta_i;
        chikk_i(:,i+M+1)=chikk(:,k)- rho*chol(uPakk(:,:,k))*zeta_i;
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
        %E_i_kk(:,p)=chikk_i(:,p) - chikk(:,k);  % for the calculation of Pa matrix
        % covariances calculation
        PEEkk1 = PEEkk1 + omega_i(:,p)*E_i(:,p)*E_i(:,p)'; 
        PEekk= PEekk + omega_i(:,p)*E_i(:,p)*e_i(:,p)';
        Peekk= Peekk + omega_i(:,p)*e_i(:,p)*e_i(:,p)';
        %Pakk= Pakk + omega_i(:,p)*E_i_kk(:,p)*E_i_kk(:,p)';
    end
    %Kalman Gain Update     
    Lk=PEekk*inv(Peekk);
    e_u(:,k)= ey(:,k) - Yhat_kk1(:,k) ; % innovation ---- measurement variable ey is taken from above (near EKF)
    xhat_kk(:,k+1)=xhat_kk1(:,k) + Lk*e_u(:,k) ; % state update
    uPkk(:,:,k+1) = PEEkk1 - Lk*Peekk*Lk';  %covariance update
    uPakk(:,:,k+1)=blkdiag(uPkk(:,:,k+1),Q,R); % the big combined covariance matrix update
    chikk(:,k+1)= [xhat_kk(:,k+1);mw';mv'];
    chikk_i(:,1)=chikk(:,k+1); % first element of chi updated
    truPkk(k)=trace(uPkk(:,:,k)); %trace of Pkk
    spec_rad_UKF_Pkk(k)=max(abs(eig(uPkk(:,:,k)))); % spectral radii of updated covariance
    spec_rad_UKF_Pkk1(k)=max(abs(eig(PEEkk1))); % spectral radii of predicted covariance
    betak_UKF(:,k)=(x(:,k)-xhat_kk(:,k))'*inv(uPkk(:,:,k))*(x(:,k)-xhat_kk(:,k));% NESS calculation
end
%% plots for UKF
figure(3)
subplot(321),plot(T,x(1,:),T,xhat_kk(1,:)), ylabel('x_1 (flux)'), title('UKF'),legend('true', 'estimated')
subplot(322),plot(T,x(2,:),T,xhat_kk(2,:)), ylabel('x_2 (flux)'), title('UKF'),legend('true', 'estimated')
subplot(323),plot(T,x(3,:),T,xhat_kk(3,:)), ylabel('x_3 (flux)'), title('UKF'),legend('true', 'estimated')
subplot(324),plot(T,x(4,:),T,xhat_kk(4,:)), ylabel('x_4 (flux)'), title('UKF'),legend('true', 'estimated')
subplot(325),plot(T,x(5,:),T,xhat_kk(5,:)), ylabel('x_5 (angular velocity)'), title('UKF'),legend('true', 'estimated')
close
%% plotting of true, KF, EKF, and UKF
figure(4)
subplot(321),plot(T,x(1,:),T,xkk(1,:),T,exkk(1,:),T,xhat_kk(1,:)), ylabel('x_1 (flux)'), title('True vs estimated'),legend('true', 'KF','EKF','UKF')
subplot(322),plot(T,x(2,:),T,xkk(2,:),T,exkk(2,:),T,xhat_kk(2,:)), ylabel('x_2 (flux)'), title('True vs estimated'),legend('true', 'KF','EKF','UKF')
subplot(323),plot(T,x(3,:),T,xkk(3,:),T,exkk(3,:),T,xhat_kk(3,:)), ylabel('x_3 (flux)'), title('True vs estimated'),legend('true', 'KF','EKF','UKF')
subplot(324),plot(T,x(4,:),T,xkk(4,:),T,exkk(4,:),T,xhat_kk(4,:)), ylabel('x_4 (flux)'), title('True vs estimated'),legend('true', 'KF','EKF','UKF')
subplot(325),plot(T,x(5,:),T,xkk(5,:),T,exkk(5,:),T,xhat_kk(5,:)), ylabel('x_5 (angular velocity)'), title('True vs estimated'),legend('true', 'KF','EKF','UKF')
saveas(figure(4),'plots/true_vs_estimated_all.png')
%% plotting of true, EKF and UKF on same plot
figure(5); suptitle('True Vs Estimated');
subplot(321),plot(T,x(1,:),T,exkk(1,:),T,xhat_kk(1,:)),xlabel('Sampling instants'), ylabel('x_1 (flux)'),legend({'true', 'EKF','UKF'})%,'FontSize',7)
subplot(322),plot(T,x(2,:),T,exkk(2,:),T,xhat_kk(2,:)),xlabel('Sampling instants'), ylabel('x_2 (flux)'),legend('true', 'EKF','UKF')
subplot(323),plot(T,x(3,:),T,exkk(3,:),T,xhat_kk(3,:)),xlabel('Sampling instants'), ylabel('x_3 (flux)'),legend('true', 'EKF','UKF')
subplot(324),plot(T,x(4,:),T,exkk(4,:),T,xhat_kk(4,:)),xlabel('Sampling instants'), ylabel('x_4 (flux)'), legend('true','EKF','UKF')
subplot(325),plot(T,x(5,:),T,exkk(5,:),T,xhat_kk(5,:)),xlabel('Sampling instants'), ylabel('x_5 (angular velocity)'),legend('true','EKF','UKF')
saveas(figure(5),'plots/true_vs_estimated_noKF.png')
%% innovation plot for KF,EKF and UKF
figure(6);suptitle('Innovation')
subplot(211),plot(T(2:end),e(1,:),T(2:end),E(1,:),T(2:end),e_u(1,:)),xlabel('Sampling instants'), ylabel('y_1'),legend( 'KF','EKF','UKF')
subplot(212),plot(T(2:end),e(2,:),T(2:end),E(2,:),T(2:end),e_u(2,:)),xlabel('Sampling instants'), ylabel('y_2'),legend('KF', 'EKF','UKF')
saveas(figure(6),'plots/innovation_all.png')
%% innovation plot for EKF and UKF
figure(7);suptitle('Innovation')
subplot(211),plot(T(2:end),E(1,:),T(2:end),e_u(1,:)),xlabel('Sampling instants'), ylabel('y_1'),legend( 'EKF','UKF')
subplot(212),plot(T(2:end),E(2,:),T(2:end),e_u(2,:)),xlabel('Sampling instants'), ylabel('y_2'),legend( 'EKF','UKF')
saveas(figure(7),'plots/innovation_noKF.png')
%% Plot of Spectral radii for KF, EKF and UKF
figure(8)
plot(T(2:end),spec_rad_KF_Pkk,T(2:end),spec_rad_KF_Pkk1,T(2:end),spec_rad_EKF_Pkk,T(2:end),spec_rad_EKF_Pkk1,T(2:end),spec_rad_UKF_Pkk,T(2:end), spec_rad_UKF_Pkk1)
legend('spec radii KF update', 'spec radii KF predicted','spec radii EKF update', 'spec radii EKF predicted','spec radii UKF update', 'spec radii UKF predicted')
ylabel('spectral radii');xlabel('Sampling instants'); title('Spectral radii of predicted and updated covariances from various filters');
saveas(figure(8),'plots/spectral_radii_all.png')
% separately plotted spectral radii
figure(9)
subplot(311), plot(T(2:end),spec_rad_KF_Pkk,T(2:end),spec_rad_KF_Pkk1),legend('spec radii updated', 'spec radii predicted'),title('KF');
subplot(312), plot(T(2:end),spec_rad_EKF_Pkk,T(2:end),spec_rad_EKF_Pkk1),legend('spec radii updated', 'spec radii predicted'),title('EKF');
subplot(313), plot(T(2:end),spec_rad_UKF_Pkk,T(2:end),spec_rad_UKF_Pkk1),legend('spec radii updated', 'spec radii predicted'),title('UKF');
saveas(figure(9),'plots/spectral_radii_all_subplots.png')
%% Plot of estimation error for KF, EKF, UKF
% calculation of 3 standard deviation bounds
for k =1:N+1
    cov_e=(x(:,k)-xkk(:,k))*(x(:,k)-xkk(:,k))'; %covariance of error for KF
    std_KF(:,k)=3*sqrt(diag(cov_e));
    cov_e=(x(:,k)-exkk(:,k))*(x(:,k)-exkk(:,k))';% covariance of error of EKF
    std_EKF(:,k)=3*sqrt(diag(cov_e));
    cov_e=(x(:,k)-xhat_kk(:,k))*(x(:,k)-xhat_kk(:,k))'; % covariance of error for UKF
    std_UKF(:,k)=3*sqrt(diag(cov_e));
end
figure(10); suptitle('Estimation error')
subplot(321),plot(T,x(1,:)-xkk(1,:),T,x(1,:)-exkk(1,:),T,x(1,:)-xhat_kk(1,:)), ylabel('x_1'),xlabel('Sampling instatns'),legend('KF','EKF','UKF')
subplot(322),plot(T,x(2,:)-xkk(1,:),T,x(2,:)-exkk(2,:),T,x(2,:)-xhat_kk(2,:)), ylabel('x_2'),xlabel('Sampling instatns'),legend('KF','EKF','UKF')
subplot(323),plot(T,x(3,:)-xkk(1,:),T,x(3,:)-exkk(3,:),T,x(3,:)-xhat_kk(3,:)), ylabel(' x_3'),xlabel('Sampling instatns'),legend('KF','EKF','UKF')
subplot(324),plot(T,x(4,:)-xkk(1,:),T,x(4,:)-exkk(4,:),T,x(4,:)-xhat_kk(4,:)), ylabel('x_4'),xlabel('Sampling instatns'),legend('KF','EKF','UKF')
subplot(325),plot(T,x(5,:)-xkk(1,:),T,x(5,:)-exkk(5,:),T,x(5,:)-xhat_kk(5,:)), ylabel('x_5'),xlabel('Sampling instatns'),legend('KF','EKF','UKF')
saveas(figure(10),'plots/estimation_error_all.png')
%% Plot of estimation error for EKF and UKF along with standard deviation bounds
ttl='Estimation error for filters along with \pm 3 standard deviation bounds';
figure(11);plot(T,x(1,:)-exkk(1,:),T,std_EKF(1,:),T,-std_EKF(1,:),T,x(1,:)-xhat_kk(1,:),T,std_UKF(1,:),T,-std_UKF(1,:)), ylabel('x_1'),title(ttl),xlabel('Sampling instatns'),legend('EKF','+3\sigma EKF','-3\sigma EKF','UKF','+3\sigma UKF','-3\sigma UKF');saveas(figure(11),'plots/estimation_error_noKF_sigma_s1.png')
figure(12);plot(T,x(2,:)-exkk(2,:),T,std_EKF(2,:),T,-std_EKF(2,:),T,x(2,:)-xhat_kk(2,:),T,std_UKF(2,:),T,-std_UKF(2,:)), ylabel('x_2'),title(ttl),xlabel('Sampling instatns'),legend('EKF','+3\sigma EKF','-3\sigma EKF','UKF','+3\sigma UKF','-3\sigma UKF');saveas(figure(12),'plots/estimation_error_noKF_sigma_s2.png')
figure(13);plot(T,x(3,:)-exkk(3,:),T,std_EKF(3,:),T,-std_EKF(3,:),T,x(3,:)-xhat_kk(3,:),T,std_UKF(3,:),T,-std_UKF(3,:)), ylabel('x_3'),title(ttl),xlabel('Sampling instatns'),legend('EKF','+3\sigma EKF','-3\sigma EKF','UKF','+3\sigma UKF','-3\sigma UKF');saveas(figure(13),'plots/estimation_error_noKF_sigma_s3.png')
figure(14);plot(T,x(4,:)-exkk(4,:),T,std_EKF(4,:),T,-std_EKF(4,:),T,x(4,:)-xhat_kk(4,:),T,std_UKF(4,:),T,-std_UKF(4,:)), ylabel('x_4'),title(ttl),xlabel('Sampling instatns'),legend('EKF','+3\sigma EKF','-3\sigma EKF','UKF','+3\sigma UKF','-3\sigma UKF');saveas(figure(14),'plots/estimation_error_noKF_sigma_s4.png')
figure(15);plot(T,x(5,:)-exkk(5,:),T,std_EKF(5,:),T,-std_EKF(5,:),T,x(5,:)-xhat_kk(5,:),T,std_UKF(5,:),T,-std_UKF(5,:)), ylabel('x_5'),title(ttl),xlabel('Sampling instatns'),legend('EKF','+3\sigma EKF','-3\sigma EKF','UKF','+3\sigma UKF','-3\sigma UKF');saveas(figure(15),'plots/estimation_error_noKF_sigma_s5.png')
%% Plot of estimation error for EKF along with standard deviation bounds
figure(16);suptitle('Estimation error for EKF along with \pm 3 standard deviation bounds')
subplot(321),plot(T,x(1,:)-exkk(1,:),T,std_EKF(1,:),T,-std_EKF(1,:)), ylabel('x_1'),xlabel('Sampling instatns'),legend('EKF','+3\sigma EKF','-3\sigma EKF')
subplot(322),plot(T,x(2,:)-exkk(2,:),T,std_EKF(2,:),T,-std_EKF(2,:)), ylabel('x_2'),xlabel('Sampling instatns'),legend('EKF','+3\sigma EKF','-3\sigma EKF')
subplot(323),plot(T,x(3,:)-exkk(3,:),T,std_EKF(3,:),T,-std_EKF(3,:)), ylabel('x_3'),xlabel('Sampling instatns'),legend('EKF','+3\sigma EKF','-3\sigma EKF')
subplot(324),plot(T,x(4,:)-exkk(4,:),T,std_EKF(4,:),T,-std_EKF(4,:)), ylabel('x_4'),xlabel('Sampling instatns'),legend('EKF','+3\sigma EKF','-3\sigma EKF')
subplot(325),plot(T,x(5,:)-exkk(5,:),T,std_EKF(5,:),T,-std_EKF(5,:)), ylabel('x_5'),xlabel('Sampling instatns'),legend('EKF','+3\sigma EKF','-3\sigma EKF')
saveas(figure(16),'plots/estimation_error_EKF_sigma_all_states.png')
%% Plot of estimation error for UKF along with standard deviation bounds
figure(17);suptitle('Estimation error for UKF along with \pm 3 standard deviation bounds')
subplot(321),plot(T,x(1,:)-xhat_kk(1,:),T,std_UKF(1,:),T,-std_UKF(1,:)), ylabel('x_1'),xlabel('Sampling instatns'),legend({'UKF','+3\sigma UKF','-3\sigma UKF'})%,'Location','NorthEastOutside')
subplot(322),plot(T,x(2,:)-xhat_kk(2,:),T,std_UKF(2,:),T,-std_UKF(2,:)), ylabel('x_2'),xlabel('Sampling instatns'),legend({'UKF','+3\sigma UKF','-3\sigma UKF'})%,'Location','NorthEastOutside')
subplot(323),plot(T,x(3,:)-xhat_kk(3,:),T,std_UKF(3,:),T,-std_UKF(3,:)), ylabel('x_3'),xlabel('Sampling instatns'),legend({'UKF','+3\sigma UKF','-3\sigma UKF'})%,'Location','NorthEastOutside')
subplot(324),plot(T,x(4,:)-xhat_kk(4,:),T,std_UKF(4,:),T,-std_UKF(4,:)), ylabel('x_4'),xlabel('Sampling instatns'),legend({'UKF','+3\sigma UKF','-3\sigma UKF'})%,'Location','NorthEastOutside')
subplot(325),plot(T,x(5,:)-xhat_kk(5,:),T,std_UKF(5,:),T,-std_UKF(5,:)), ylabel('x_5'),xlabel('Sampling instatns'),legend({'UKF','+3\sigma UKF','-3\sigma UKF'})%,'Location','NorthEastOutside')
saveas(figure(17),'plots/estimation_error_UKF_sigma_all_states.png')

%% mean and covariance of each innovation %table1
name = {'Mean KF';'Var KF';'Mean EKF';'Var EKF';'Mean UKF'; 'Var UKF'};
y1=[mean(e(1,:));var(e(1,:));mean(E(1,:));var(E(1,:));mean(e_u(1,:));var(e_u(1,:))];
y2=[mean(e(2,:));var(e(2,:));mean(E(2,:));var(E(2,:));mean(e_u(2,:));var(e_u(2,:))];
table = table(name,y1,y2) % for making table of mean and variances
%% RMSE calculations % table2
for i=1:5
    rmse_KF(i)=sqrt(mean((x(i,:) - xkk(i,:)).^2)) ;
    rmse_EKF(i)=sqrt(mean((x(i,:) - exkk(i,:)).^2)) ;
    rmse_UKF(i)=sqrt(mean((x(i,:) - xhat_kk(i,:)).^2)) ;
end
disp('RMSE for KF for all states respectively '); rmse_KF
disp('RMSE for EKF for all states respectively '); rmse_EKF
disp('RMSE for UKF for all states respectively '); rmse_UKF
%% NESS and chi square part
n=5; alpha = 0.05
zeta1=chi2inv(alpha,n); zeta2=chi2inv(1-alpha,n)
% betak plot for all filter
figure(18);suptitle('\beta_k for kF,EKF and UKF')
plot(T(2:end),betak_KF,T(2:end),betak_EKF,T(2:end),betak_UKF,T(2:end),zeta1*ones(1,N),T(2:end),zeta2*ones(1,N)),legend('KF','EKF', 'UKF','zeta1','zeta2')
xlabel('Sampling instants');ylabel('\beta_k');saveas(figure(18),'plots/betak_all.png')
% betak plot for EKF and UKF together
figure(19);suptitle('\beta_k for EKF and UKF')
plot(T(2:end),betak_EKF,T(2:end),betak_UKF,T(2:end),zeta1*ones(1,N),T(2:end),zeta2*ones(1,N)),legend('EKF', 'UKF','zeta1','zeta2')
xlabel('Sampling instants');ylabel('\beta_k');saveas(figure(19),'plots/betak_noKF.png')
% betak plot for EKF only
figure(20);suptitle('\beta_k for EKF')
plot(T(2:end),betak_EKF,T(2:end),zeta1*ones(1,N),T(2:end),zeta2*ones(1,N)),legend('EKF','zeta1','zeta2')
xlabel('Sampling instants');ylabel('\beta_k');saveas(figure(20),'plots/betak_EKF.png')
%% for computing fraction of time instants betak exceeded the bond
countKF=0;countEKF=0;countUKF=0;
for k=1:N
    if betak_KF(k)<=zeta1 ||   betak_KF(k)>=zeta2 % KF condition for out of bound
        countKF=countKF+1;
    end
    if betak_EKF(k)<=zeta1 ||   betak_EKF(k)>=zeta2 %EKF  condition for out of bound
        countEKF=countEKF+1;
    end
    if betak_UKF(k)<=zeta1 ||   betak_UKF(k)>=zeta2 % UKF condition for out of bound
        countUKF=countUKF+1;
    end
end
fracKF=1.0*countKF/N
fracEKF=1.0*countEKF/N
fracUKF=1.0*countUKF/N

