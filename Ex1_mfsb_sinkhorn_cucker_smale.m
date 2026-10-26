clear all;
close all;
clc;

%% 
sigma2 = 1;
D = sigma2/2;

xMin = -3; xMax = 3;
vMin = -6; vMax = 6;

Nx = 120;
Nv = 241;
Nt = 1000;


Lx = xMax - xMin;
x  = linspace(xMin,xMax,Nx+1)';
x(end) = [];
v  = linspace(vMin,vMax,Nv)';
dx = Lx/Nx;
dv = v(2)-v(1);

t  = linspace(0,1,Nt+1);
dt = 1/Nt;

ItersOut = 50;
ItersMid = 150;
tolMid = 1e-5;
tolOut = 1e-5;

maxSBInit = 150;
tolSBInit = 1e-5;



bcVarphi = 1e-16;
bcHat    = 1e-16;
bcMu     = 0;
posTol   = 0;
thr      = 1e-8;
theta    = 1;
clipH    = 1;

%% Endpoint specifications
phaseFloor = 1e-14;
distT = @(c) mod(x - c + 0.5*Lx,Lx) - 0.5*Lx;

x0Center = 0.0;
sx0      = 0.35;

vLeft0   = -1.5;
vRight0  = 1;
sv0l     = 0.4;
sv0r     = 0.4;

rho0 = exp(-0.5*(distT(x0Center)/sx0).^2);

Mleft0  = exp(-0.5*((v - vLeft0 )/sv0l).^2);
Mright0 = exp(-0.5*((v - vRight0)/sv0r).^2);

mu_in_raw = rho0 * (0.5*Mleft0 + 0.5*Mright0).';

xLeftT  = -0.8;
xRightT =  0.8;
sxT     =  0.25;

vT  = 0.0;
svT = 0.2;

rhoLeftT  = exp(-0.5*(distT(xLeftT )/sxT).^2);
rhoRightT = exp(-0.5*(distT(xRightT)/sxT).^2);

MT = exp(-0.5*((v - vT)/svT).^2);

mu_fin_raw = (0.4*rhoLeftT + 0.6*rhoRightT) * MT.';


mu_in_raw(:,2:Nv-1)  = max(mu_in_raw(:,2:Nv-1),phaseFloor);
mu_fin_raw(:,2:Nv-1) = max(mu_fin_raw(:,2:Nv-1),phaseFloor);


mu_in  = normalize2d_vboundary(mu_in_raw,dx,dv,bcMu,posTol,'mu_in');
mu_fin = normalize2d_vboundary(mu_fin_raw,dx,dv,bcMu,posTol,'mu_fin');


rho_in  = sum(mu_in,2)*dv;
rho_fin = sum(mu_fin,2)*dv;

M_in  = sum(mu_in,1)'*dx;
M_fin = sum(mu_fin,1)'*dx;

%% Interaction Kernel
alphaCS = 0.45;
alignmentStrength = 3;

S = periodic_distance_matrix(x);
Wmat = alignmentStrength./(1 + S.^2).^alphaCS;

%% 
dtHalf = 0.5*dt;
AvHalfInt = diffusion_v_matrix(Nv,dtHalf,D,dv);
AvHalfIntDec = decomposition(AvHalfInt,'lu');
AvHalfAdjIntDec = decomposition(AvHalfInt.','lu');
[T0,~] = cs_affine_pullback(x,v,zeros(Nx,1),zeros(Nx,1),dt);

%% Initialization: Classical Sinkhorn
[mu_k,varphi_k,hatvarphi_k,errSBInit,lastSBInit,errPhi1Init,errHat1Init] = strang_SB_mu_endpoint( ...
    mu_in,mu_fin,dv,dt,D,Nt,AvHalfIntDec,AvHalfAdjIntDec,T0, ...
    bcVarphi,bcHat,bcMu,posTol,thr,tolSBInit,maxSBInit);

muSB = mu_k;
varphiSB = varphi_k;
hatvarphiSB = hatvarphi_k;
fprintf('noninteracting kinetic SB init | iters %d | err %.3e | errPhi1 %.3e | errHat1 %.3e | endpoint0 %.3e | endpointT %.3e\n', ...
    lastSBInit,errSBInit(max(lastSBInit,1)), ...
    errPhi1Init(max(lastSBInit,1)), ...
    errHat1Init(max(lastSBInit,1)), ...
    hilbert_metric(muSB(:,:,1),mu_in,thr,clipH), ...
    hilbert_metric(muSB(:,:,end),mu_fin,thr,clipH));

%% Generalized Sinkhorn
errOut = nan(ItersOut,1);
errMid = nan(ItersMid,1);
massErr = nan(ItersOut,1);

for k = 1:ItersOut
    rho_k = zeros(Nx,Nt+1);
    momentum_k = zeros(Nx,Nt+1);
    alignRate_k = zeros(Nx,Nt+1);
    weightedMomentum_k = zeros(Nx,Nt+1);

    for n = 1:Nt+1
        [rho_k(:,n),momentum_k(:,n),alignRate_k(:,n),weightedMomentum_k(:,n)] = ...
            cs_force_coefficients(mu_k(:,:,n),v,Wmat,dx,dv);
    end


    alignRateHalf_k = 0.5*(alignRate_k(:,1:Nt)+alignRate_k(:,2:Nt+1));
    weightedMomentumHalf_k = 0.5*(weightedMomentum_k(:,1:Nt)+weightedMomentum_k(:,2:Nt+1));

    T_k = cell(Nt,1);
    clipMask_k = cell(Nt,1);
    for n = 1:Nt
        [T_k{n},clipMask_k{n}] = cs_affine_pullback( ...
            x,v,alignRateHalf_k(:,n),weightedMomentumHalf_k(:,n),dt);
    end

    maxClipMass = 0;
    for n = 1:Nt
        clippedMass_n = sum(mu_k(:,:,n+1).*clipMask_k{n},'all')*dx*dv;
        maxClipMass = max(maxClipMass,clippedMass_n);
    end


    fprintf('max clipped mass k=%d = %.3e\n', k, maxClipMass);

    varphi_j    = varphi_k;
    hatvarphi_j = hatvarphi_k;
    errMid(:) = nan;

    for j = 1:ItersMid

        qSlope_j = zeros(Nx,Nt+1);
        qIntercept_j = zeros(Nx,Nt+1);
        ivQ = 3:Nv-2;
        for n = 1:Nt+1
            dvarphi_dv = centered_Dv_2D(varphi_j(:,:,n),dv);
            [qSlope_j(:,n),qIntercept_j(:,n)] = cs_q_coefficients( ...
                hatvarphi_j(:,:,n),dvarphi_dv,v,Wmat,dx,dv,ivQ);
        end

        varphi_next = bcVarphi*ones(Nx,Nv,Nt+1);
        varphi_next(:,:,end) = impose_phase_boundary(varphi_j(:,:,end),bcVarphi);
        for n = Nt+1:-1:2
            ell = n-1;
            qSlopeHalf = 0.5*(qSlope_j(:,ell)+qSlope_j(:,ell+1));
            qInterceptHalf = 0.5*(qIntercept_j(:,ell)+qIntercept_j(:,ell+1));

       
            Ghalf = cs_reaction_matrix(qSlopeHalf,qInterceptHalf, ...
                alignRateHalf_k(:,ell),v);
%Backward Eq.
            varphi_next(:,:,n-1) = strang_adj_interacting( ...
                varphi_next(:,:,n),AvHalfAdjIntDec,T_k{ell},Ghalf,dt,D,dv,bcVarphi,posTol, ...
                sprintf('varphi adjoint k=%d,j=%d,n=%d',k,j,n));
        end

        hatvarphi_next = bcHat*ones(Nx,Nv,Nt+1);
        hatvarphi_next(:,:,1) = divide_on_interior(mu_in,varphi_next(:,:,1),bcHat,posTol, ...
            sprintf('hatvarphi0 k=%d,j=%d',k,j));


        for n = 1:Nt
            qSlopeHalf = 0.5*(qSlope_j(:,n)+qSlope_j(:,n+1));
            qInterceptHalf = 0.5*(qIntercept_j(:,n)+qIntercept_j(:,n+1));
            Ghalf = cs_reaction_matrix(qSlopeHalf,qInterceptHalf, ...
                alignRateHalf_k(:,n),v);
%Forward Eq.
            hatvarphi_next(:,:,n+1) = strang_interacting( ...
                hatvarphi_next(:,:,n),AvHalfIntDec,T_k{n},Ghalf,dt,D,dv,bcHat,posTol, ...
                sprintf('hatvarphi forward k=%d,j=%d,n=%d',k,j,n));
        end

        varphi_next(:,:,end) = divide_on_interior(mu_fin,hatvarphi_next(:,:,end),bcVarphi,posTol, ...
            sprintf('varphi1 k=%d,j=%d',k,j));

        err = 0;
        for n = 1:Nt+1
            err = max(err,hilbert_metric(varphi_next(:,:,n),varphi_j(:,:,n),thr,clipH));
            err = max(err,hilbert_metric(hatvarphi_next(:,:,n),hatvarphi_j(:,:,n),thr,clipH));
        end
        errMid(j) = err;

        varphi_j = varphi_next;
        hatvarphi_j = hatvarphi_next;

        if errMid(j) < tolMid
            break;
        end
    end

    varphi_kp1 = varphi_j;
    hatvarphi_kp1 = hatvarphi_j;

    mu_kp1 = bcMu*ones(Nx,Nv,Nt+1);
    massTime = zeros(Nt+1,1);
    for n = 1:Nt+1
        cand = reconstruct_density(varphi_kp1(:,:,n),hatvarphi_kp1(:,:,n),bcMu,posTol, ...
            sprintf('candidate k=%d,n=%d',k,n));
        massTime(n) = sum(cand,'all')*dx*dv;
        mu_kp1(:,:,n) = damp_density(cand,mu_k(:,:,n),theta,bcMu,posTol, ...
            sprintf('damped mu k=%d,n=%d',k,n));
    end

    massErr(k) = max(abs(massTime - 1));
    err = 0;
    for n = 1:Nt+1
        err = max(err,hilbert_metric(mu_kp1(:,:,n),mu_k(:,:,n),thr,clipH));
    end
    errOut(k) = err;

    endpointErr0 = hilbert_metric(mu_kp1(:,:,1),mu_in,thr,clipH);
    endpointErr1 = hilbert_metric(mu_kp1(:,:,end),mu_fin,thr,clipH);

    fprintf('outer %3d | mid %3d | errOut %.3e | errMid %.3e | massErr %.3e | end0 %.3e | endT %.3e\n', ...
        k,j,errOut(k),errMid(j),massErr(k),endpointErr0,endpointErr1);

    mu_k = mu_kp1;
    varphi_k = varphi_kp1;
    hatvarphi_k = hatvarphi_kp1;

    if errOut(k) < tolOut
        break;
    end
end

mu = mu_k;
varphi = varphi_k;
hatvarphi = hatvarphi_k;
lastOuter = k;
errOutPlot = errOut(1:lastOuter);

%% Save Cucker-Smale-interaction results
resultsFolder = fullfile(pwd,'saved_results');

if ~exist(resultsFolder,'dir')
    mkdir(resultsFolder);
end

resultsFile = fullfile( ...
    resultsFolder, ...
    'Ex1_CuckerSmale_full_results.mat');

%% 
params = struct;

params.sigma2 = sigma2;
params.D      = D;

params.xMin = xMin;
params.xMax = xMax;
params.vMin = vMin;
params.vMax = vMax;

params.Nx = Nx;
params.Nv = Nv;
params.Nt = Nt;

params.dx = dx;
params.dv = dv;
params.dt = dt;

params.alphaCS  = alphaCS;
params.alignmentStrength = alignmentStrength;

params.bcVarphi = bcVarphi;
params.bcHat    = bcHat;
params.bcMu     = bcMu;
params.posTol   = posTol;

params.ItersOut = ItersOut;
params.ItersMid = ItersMid;
params.tolMid   = tolMid;
params.tolOut   = tolOut;

params.maxSBInit = maxSBInit;
params.tolSBInit = tolSBInit;

params.theta = theta;
params.thr   = thr;
params.clipH = clipH;

params.phaseFloor = phaseFloor;


rho = squeeze(sum(mu,2))*dv;
M   = squeeze(sum(mu,1))*dx;


massTrajectory = squeeze(sum(sum(mu,1),2))*dx*dv;
massTrajectory = massTrajectory(:);

%% 
diagnostics = struct;

diagnostics.lastOuter = lastOuter;
diagnostics.lastMid   = j;

diagnostics.errOut  = errOut(1:lastOuter);
diagnostics.massErr = massErr(1:lastOuter);

diagnostics.finalErrOut = errOut(lastOuter);
diagnostics.finalErrMid = errMid(j);

diagnostics.converged = ...
    diagnostics.finalErrOut < tolOut;

% Full phase-space endpoint Hilbert errors
diagnostics.endpointError0 = hilbert_metric( ...
    mu(:,:,1),mu_in,thr,clipH);

diagnostics.endpointErrorT = hilbert_metric( ...
    mu(:,:,end),mu_fin,thr,clipH);

% Full phase-space endpoint L1 errors
diagnostics.endpointL1Error0 = ...
    sum(abs(mu(:,:,1)-mu_in),'all')*dx*dv;

diagnostics.endpointL1ErrorT = ...
    sum(abs(mu(:,:,end)-mu_fin),'all')*dx*dv;

% Spatial marginal endpoint L1 errors
diagnostics.rhoEndpointError0 = ...
    sum(abs(rho(:,1)-rho_in))*dx;

diagnostics.rhoEndpointErrorT = ...
    sum(abs(rho(:,end)-rho_fin))*dx;

% Velocity marginal endpoint L1 errors
diagnostics.MEndpointError0 = ...
    sum(abs(M(:,1)-M_in))*dv;

diagnostics.MEndpointErrorT = ...
    sum(abs(M(:,end)-M_fin))*dv;

% Mass diagnostics
diagnostics.massTrajectory = massTrajectory;
diagnostics.maxMassErrorFinal = ...
    max(abs(massTrajectory-1));

% Noninteracting initialization diagnostics
diagnostics.lastSBInit = lastSBInit;
diagnostics.errSBInit = errSBInit(1:lastSBInit);
diagnostics.errPhi1Init = errPhi1Init(1:lastSBInit);
diagnostics.errHat1Init = errHat1Init(1:lastSBInit);

diagnostics.SBEndpointError0 = hilbert_metric( ...
    muSB(:,:,1),mu_in,thr,clipH);

diagnostics.SBEndpointErrorT = hilbert_metric( ...
    muSB(:,:,end),mu_fin,thr,clipH);


%% Save variables needed for plotting
save(resultsFile, ...
    'mu', ...
    'varphi', ...
    'hatvarphi', ...
    'muSB', ...
    'varphiSB', ...
    'hatvarphiSB', ...
    'mu_in', ...
    'mu_fin', ...
    'rho_in', ...
    'rho_fin', ...
    'M_in', ...
    'M_fin', ...
    'rho', ...
    'M', ...
    'x', ...
    'v', ...
    't', ...
    'Wmat', ...
    'params', ...
    'diagnostics', ...
    '-v7.3');

%%
fprintf('\nExample I results saved to:\n%s\n',resultsFile);

fprintf('\nFinal interacting diagnostics:\n');
fprintf('outer iterations                    = %d\n', ...
    diagnostics.lastOuter);

fprintf('final outer error                   = %.3e\n', ...
    diagnostics.finalErrOut);

fprintf('final inner error                   = %.3e\n', ...
    diagnostics.finalErrMid);

fprintf('initial phase-space endpoint error  = %.3e\n', ...
    diagnostics.endpointError0);

fprintf('final phase-space endpoint error    = %.3e\n', ...
    diagnostics.endpointErrorT);

fprintf('initial phase-space L1 error        = %.3e\n', ...
    diagnostics.endpointL1Error0);

fprintf('final phase-space L1 error          = %.3e\n', ...
    diagnostics.endpointL1ErrorT);

fprintf('maximum mass error                  = %.3e\n', ...
    diagnostics.maxMassErrorFinal);
  

%% Local functions
function [mu_sb,varphi_sb,hatvarphi_sb,errSB,lastIter,errPhi1,errHat1] = strang_SB_mu_endpoint( ...
    mu_in,mu_fin,dv,dt,D,Nt,AvHalfIntDec,AvHalfAdjIntDec,T0, ...
    bcValVarphi,bcValHat,bcMu,posTol,thr,tolSB,maxIter)

    [Nx,Nv] = size(mu_in);
    varphi1 = impose_phase_boundary(ones(Nx,Nv),bcValVarphi);
    errSB   = nan(maxIter,1);
    errPhi1 = nan(maxIter,1);
    errHat1 = nan(maxIter,1);
    lastIter = maxIter;


hat1_old = [];

for iter = 1:maxIter
    u = varphi1;
    for tt = Nt+1:-1:2
        u = strang_adj_interacting(u,AvHalfAdjIntDec,T0,zeros(Nx,1),dt,D,dv,bcValVarphi,posTol,'SB adjoint');
    end

    hat0 = divide_on_interior(mu_in,u,bcValHat,posTol,'SB hat0');

    uhat = hat0;
    for tt = 1:Nt
        uhat = strang_interacting(uhat,AvHalfIntDec,T0,zeros(Nx,1),dt,D,dv,bcValHat,posTol,'SB forward');
    end

    varphi1_next = divide_on_interior(mu_fin,uhat,bcValVarphi,posTol,'SB varphi1');

    errPhi1(iter) = hilbert_metric(varphi1,varphi1_next,thr,2);

    if isempty(hat1_old)
        errHat1(iter) = Inf;
    else
        errHat1(iter) = hilbert_metric(hat1_old,uhat,thr,2);
    end

    errSB(iter) = max(errPhi1(iter),errHat1(iter));

    varphi1 = varphi1_next;
    hat1_old = uhat;

    if errSB(iter) < tolSB
        lastIter = iter;
        break;
    end
end

    varphi_sb = bcValVarphi*ones(Nx,Nv,Nt+1);
    varphi_sb(:,:,end) = impose_phase_boundary(varphi1,bcValVarphi);
    for tt = Nt+1:-1:2
        varphi_sb(:,:,tt-1) = strang_adj_interacting(varphi_sb(:,:,tt),AvHalfAdjIntDec,T0,zeros(Nx,1),dt,D,dv,bcValVarphi,posTol,'SB traj adjoint');
    end

    hatvarphi_sb = bcValHat*ones(Nx,Nv,Nt+1);
    hatvarphi_sb(:,:,1) = divide_on_interior(mu_in,varphi_sb(:,:,1),bcValHat,posTol,'SB traj hat0');
    for tt = 1:Nt
        hatvarphi_sb(:,:,tt+1) = strang_interacting(hatvarphi_sb(:,:,tt),AvHalfIntDec,T0,zeros(Nx,1),dt,D,dv,bcValHat,posTol,'SB traj forward');
    end

    mu_sb = bcMu*ones(Nx,Nv,Nt+1);
    for tt = 1:Nt+1
        mu_sb(:,:,tt) = reconstruct_density(varphi_sb(:,:,tt),hatvarphi_sb(:,:,tt),bcMu,posTol,sprintf('SB mu tt=%d',tt));
    end
end

function Uout = strang_interacting(U,AvHalfIntDec,T,G,dt,D,dv,bcVal,posTol,name)
    U = impose_phase_boundary(U,bcVal);
    require_positive_phase(U,[name ' input'],posTol);

    React = exp(-0.5*dt*G);
    U = multiply_phase(U,React);
    U = implicit_v(U,AvHalfIntDec,0.5*dt,D,dv,bcVal,posTol,[name ' diffusion half 1']);
    U = reshape(T*U(:),size(U));
    U = impose_phase_boundary(U,bcVal);
    require_positive_phase(U,[name ' transport'],posTol);
    U = implicit_v(U,AvHalfIntDec,0.5*dt,D,dv,bcVal,posTol,[name ' diffusion half 2']);
    Uout = multiply_phase(U,React);
    Uout = impose_phase_boundary(Uout,bcVal);
    require_positive_phase(Uout,[name ' output'],posTol);
end

function Uprev = strang_adj_interacting(U,AvHalfAdjIntDec,T,G,dt,D,dv,bcVal,posTol,name)
    U = impose_phase_boundary(U,bcVal);
    require_positive_phase(U,[name ' input'],posTol);

    React = exp(-0.5*dt*G);
    Y = multiply_phase(U,React);
    Y = implicit_v(Y,AvHalfAdjIntDec,0.5*dt,D,dv,bcVal,posTol,[name ' adj diffusion half 1']);
    Y = reshape(T.'*Y(:),size(U));
    Y = impose_phase_boundary(Y,bcVal);
    require_positive_phase(Y,[name ' adj transport'],posTol);
    Y = implicit_v(Y,AvHalfAdjIntDec,0.5*dt,D,dv,bcVal,posTol,[name ' adj diffusion half 2']);
    Uprev = multiply_phase(Y,React);
    Uprev = impose_phase_boundary(Uprev,bcVal);
    require_positive_phase(Uprev,[name ' output'],posTol);
end

function U = multiply_phase(U,R)
    if isvector(R) && numel(R) == size(U,1)
        U = U.*R(:);
    elseif isequal(size(R),size(U))
        U = U.*R;
    else
        error('Reaction multiplier has incompatible size.');
    end
end

function Uout = implicit_v(rhs,AvIntDec,dt,D,dv,bcVal,posTol,name)
    [Nx,Nv] = size(rhs);
    rhs = impose_phase_boundary(rhs,bcVal);
    r = dt*D/(dv^2);

    B = rhs(:,2:Nv-1);
    B(:,1) = B(:,1) + r*bcVal;
    B(:,end) = B(:,end) + r*bcVal;

    Uout = bcVal*ones(Nx,Nv);
    Uout(:,2:Nv-1) = (AvIntDec\B.').';
    Uout = impose_phase_boundary(Uout,bcVal);
    require_positive_phase(Uout,name,posTol);
end

function AvInt = diffusion_v_matrix(Nv,dt,D,dv)
    n = Nv-2;
    evec = ones(n,1);
    LvInt = spdiags([evec -2*evec evec],[-1 0 1],n,n)/(dv^2);
    AvInt = speye(n) - dt*D*LvInt;
end

function U = impose_phase_boundary(U,bcVal)
    % Periodic boundary in x: no x-row is overwritten.
    % Velocity boundary is kept at bcVal.
    U(:,1) = bcVal;
    U(:,end) = bcVal;
end

function require_positive_phase(U,name,posTol)
    if any(~isfinite(U(:)))
        error('%s has nonfinite entries.',name);
    end


    interior = U(:,2:end-1);
    if any(interior(:) <= posTol)
        [val,idx0] = min(interior(:));
        error('%s lost positivity away from v-boundary: min=%e at linear index %d.',name,val,idx0);
    end
end

function [rho,momentum,alignRate,weightedMomentum,F] = cs_force_coefficients(U,v,Wmat,dx,dv)
    % Standard one-dimensional Cucker--Smale force:
    % f[U](x,v) = (w*m)(x) - (w*rho)(x)*v.
    rho = sum(U,2)*dv;
    momentum = sum(U.*v.',2)*dv;
    alignRate = (Wmat*rho)*dx;
    weightedMomentum = (Wmat*momentum)*dx;

    if nargout >= 5
        F = weightedMomentum*ones(1,numel(v)) - alignRate*v.';
    end
end

function [qSlope,qIntercept] = cs_q_coefficients(hatvarphi,dvarphi_dv,v,Wmat,dx,dv,iv)
    % q(x,v) = -int hatphi(x~,v~) w(x-x~) (v-v~)
    %                    d_{v~}phi(x~,v~) dx~dv~
    %          = qIntercept(x) - qSlope(x)*v.
    H = hatvarphi(:,iv).*dvarphi_dv(:,iv);
    source0 = sum(H,2)*dv;
    source1 = sum(H.*v(iv).',2)*dv;
    qSlope = (Wmat*source0)*dx;
    qIntercept = (Wmat*source1)*dx;
end

function G = cs_reaction_matrix(qSlope,qIntercept,alignRate,v)
    % The forward factor equation is
    % h_t = -v h_x - f h_v + D h_vv -(q+div_v f)h.
    % In 1D, div_v f = -alignRate, hence G=q-alignRate in exp(-dt*G).
    G = (qIntercept-alignRate)*ones(1,numel(v)) - qSlope*v.';
end

function [T,clipMask] = cs_affine_pullback(x,v,alignRate,weightedMomentum,dt)
    Nx = numel(x);
    Nv = numel(v);
    dx = x(2)-x(1);
    dv = v(2)-v(1);
    Lx = Nx*dx;
    xLeft = x(1);
    N = Nx*Nv;

    clipMask = false(Nx,Nv);

    rows = zeros(4*N,1);
    cols = zeros(4*N,1);
    vals = zeros(4*N,1);
    kk = 0;

    for iv = 1:Nv
        for ix = 1:Nx
            row = ix + (iv-1)*Nx;

            % Velocity boundary is fixed.  There is no x-boundary row because x is periodic.
            if iv == 1 || iv == Nv
                kk = kk + 1;
                rows(kk) = row;
                cols(kk) = row;
                vals(kk) = 1;
                continue;
            end

            h = -dt;
            x1 = x(ix);
            v1 = v(iv);

            a1 = alignRate(ix);
            b1 = weightedMomentum(ix);
            F1 = b1 - a1*v1;

            xMid = wrap_periodic_scalar(x1 + 0.5*h*v1,xLeft,Lx);
            vMid = v1 + 0.5*h*F1;

            aMid = periodic_interp_scalar(x,alignRate,xMid);
            bMid = periodic_interp_scalar(x,weightedMomentum,xMid);
            FMid = bMid - aMid*vMid;

            x0 = wrap_periodic_scalar(x1 + h*vMid,xLeft,Lx);
            v0Raw = v1 + h*FMid;
            v0 = min(max(v0Raw,v(1)),v(end));

            if v0Raw ~= v0
                clipMask(ix,iv) = true;
            end

            sx = (x0-xLeft)/dx + 1;
            ix0 = floor(sx);
            if ix0 < 1
                ix0 = 1;
            elseif ix0 > Nx
                ix0 = Nx;
            end
            ix1 = ix0 + 1;
            if ix1 > Nx
                ix1 = 1;
            end
            ax = sx - ix0;

            sv = (v0-v(1))/dv + 1;
            iv0 = max(1,min(Nv-1,floor(sv)));
            av = sv - iv0;

            w00 = (1-ax)*(1-av);
            w10 = ax*(1-av);
            w01 = (1-ax)*av;
            w11 = ax*av;

            col00 = ix0 + (iv0-1)*Nx;
            col10 = ix1 + (iv0-1)*Nx;
            col01 = ix0 + iv0*Nx;
            col11 = ix1 + iv0*Nx;

            kk = kk + 1; rows(kk)=row; cols(kk)=col00; vals(kk)=w00;
            kk = kk + 1; rows(kk)=row; cols(kk)=col10; vals(kk)=w10;
            kk = kk + 1; rows(kk)=row; cols(kk)=col01; vals(kk)=w01;
            kk = kk + 1; rows(kk)=row; cols(kk)=col11; vals(kk)=w11;
        end
    end

    T = sparse(rows(1:kk),cols(1:kk),vals(1:kk),N,N);
end


function S = periodic_distance_matrix(x)
    % Signed shortest periodic displacement matrix S(i,j)=x_i-x_j on a torus.
    Nx = numel(x);
    dx = x(2)-x(1);
    Lx = Nx*dx;
    S = x - x.';
    S = mod(S + 0.5*Lx,Lx) - 0.5*Lx;
end

function xw = wrap_periodic_scalar(xq,xLeft,Lx)
    xw = mod(xq - xLeft,Lx) + xLeft;
end

function yq = periodic_interp_scalar(x,y,xq)
    % Linear interpolation on the periodic grid x, where x(end)+dx is x(1).
    Nx = numel(x);
    dx = x(2)-x(1);
    Lx = Nx*dx;
    xLeft = x(1);

    xq = wrap_periodic_scalar(xq,xLeft,Lx);
    sx = (xq-xLeft)/dx + 1;
    i0 = floor(sx);
    if i0 < 1
        i0 = 1;
    elseif i0 > Nx
        i0 = Nx;
    end
    i1 = i0 + 1;
    if i1 > Nx
        i1 = 1;
    end
    a = sx - i0;
    yq = (1-a)*y(i0) + a*y(i1);
end


function U = normalize2d_vboundary(U,dx,dv,bcVal,posTol,name)
    % Normalize a phase-space density with periodic x and fixed v-boundary.
    U = impose_phase_boundary(U,bcVal);
    require_positive_phase(U,name,posTol);

    boundaryMask = false(size(U));
    boundaryMask(:,1) = true;
    boundaryMask(:,end) = true;

    interiorMass = sum(U(:,2:end-1),'all')*dx*dv;
    boundaryMass = sum(U(boundaryMask),'all')*dx*dv;
    targetInteriorMass = 1 - boundaryMass;

    if targetInteriorMass <= 0 || interiorMass <= 0
        error('%s has bad v-boundary normalization.',name);
    end

    U(:,2:end-1) = U(:,2:end-1)*(targetInteriorMass/interiorMass);
    U = impose_phase_boundary(U,bcVal);
end

function U = reconstruct_density(varphi,hatvarphi,bcMu,posTol,name)
    [Nx,Nv] = size(varphi);
    if ~isequal(size(hatvarphi),[Nx,Nv])
        error('%s has incompatible factor sizes.',name);
    end

    require_positive_phase(varphi,[name ' varphi'],posTol);
    require_positive_phase(hatvarphi,[name ' hatvarphi'],posTol);

    U = bcMu*ones(Nx,Nv);
    iv = 2:Nv-1;
    U(:,iv) = varphi(:,iv).*hatvarphi(:,iv);
    U = impose_phase_boundary(U,bcMu);
    require_positive_phase(U,name,posTol);
end

function U = damp_density(cand,old,theta,bcMu,posTol,name)
    [Nx,Nv] = size(cand);
    if ~isequal(size(old),[Nx,Nv])
        error('%s has incompatible density sizes.',name);
    end

    U = bcMu*ones(Nx,Nv);
    iv = 2:Nv-1;
    U(:,iv) = theta*cand(:,iv) + (1-theta)*old(:,iv);
    U = impose_phase_boundary(U,bcMu);
    require_positive_phase(U,name,posTol);
end

function dUdv = centered_Dv_2D(U,dv)
    [Nx,Nv] = size(U);
    dUdv = zeros(Nx,Nv);
    dUdv(:,2:Nv-1) = (U(:,3:Nv)-U(:,1:Nv-2))/(2*dv);
end

function U = divide_on_interior(num,den,bcVal,posTol,name)
    [Nx,Nv] = size(num);
    U = bcVal*ones(Nx,Nv);
    idxV = 2:Nv-1;
    denInt = den(:,idxV);
    if any(~isfinite(denInt(:))) || any(denInt(:) <= 0)
        error('%s has nonpositive denominator away from v-boundary.',name);
    end
    U(:,idxV) = num(:,idxV)./denInt;
    U = impose_phase_boundary(U,bcVal);
    require_positive_phase(U,name,posTol);
end

function d = hilbert_metric(U,V,thr,clipH)
    if nargin < 4
        clipH = 0;
    end

    if isvector(U)
        U = U(:);
        V = V(:);
    else
        
        if clipH > 0 && size(U,2) > 2*clipH
            U = U(:,(1+clipH):(end-clipH));
            V = V(:,(1+clipH):(end-clipH));
        end
    end

    u = U(:);
    v = V(:);
    cond = isfinite(u) & isfinite(v) & (u > thr) & (v > thr);
    if nnz(cond) < 3
        cond = isfinite(u) & isfinite(v) & (u > 0) & (v > 0);
    end
    if nnz(cond) == 0
        d = Inf;
        return;
    end
    r = u(cond)./v(cond);
    d = log(max(r(:))/min(r(:)));
end

